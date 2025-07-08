"""
DeepStream Pipeline Configuration
Handles multi-stream video processing setup and pipeline configuration
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import sys
import configparser
import os

class DeepStreamPipeline:
    def __init__(self, config_file="config/deepstream_config.txt"):
        """Initialize DeepStream pipeline with configuration"""
        Gst.init(None)
        self.pipeline = None
        self.loop = None
        self.bus = None
        self.config_file = config_file
        self.sources = []
        self.sinks = []
        
    def create_source_bin(self, index, uri):
        """Create source bin for video input"""
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        
        # URI decode bin
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin \n")
            return None
            
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.cb_newpad, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write("Failed to add ghost pad in source bin \n")
            return None
            
        return nbin
    
    def cb_newpad(self, decodebin, decoder_src_pad, data):
        """Callback for new pad creation"""
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        
        # Check if this is video stream
        if gstname.find("video") != -1:
            bin_ghost_pad = data.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
    
    def create_pipeline(self, source_uris):
        """Create complete DeepStream pipeline"""
        # Create pipeline
        self.pipeline = Gst.Pipeline()
        
        # Create StreamMux
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write("Unable to create NvStreamMux \n")
            return False
            
        self.pipeline.add(streammux)
        
        # Add sources
        for i, uri in enumerate(source_uris):
            source_bin = self.create_source_bin(i, uri)
            if source_bin:
                self.pipeline.add(source_bin)
                padname = f"sink_{i}"
                sinkpad = streammux.get_request_pad(padname)
                if not sinkpad:
                    sys.stderr.write("Unable to create sink pad \n")
                    return False
                    
                srcpad = source_bin.get_static_pad("src")
                if not srcpad:
                    sys.stderr.write("Unable to create src pad \n")
                    return False
                    
                srcpad.link(sinkpad)
                self.sources.append(source_bin)
        
        # Set streammux properties
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', len(source_uris))
        streammux.set_property('batched-push-timeout', 4000000)
        
        # Create Primary GIE (Inference Engine)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write("Unable to create pgie \n")
            return False
            
        # Configure primary inference
        pgie.set_property('config-file-path', 'config/yolo_config.txt')
        
        # Create nvtracker
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            sys.stderr.write("Unable to create tracker \n")
            return False
            
        # Set tracker properties
        config = configparser.ConfigParser()
        config.read('config/tracker_config.txt')
        for key in config['tracker']:
            tracker.set_property(key, config.get('tracker', key))
        
        # Create tiler for multi-stream display
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            sys.stderr.write("Unable to create tiler \n")
            return False
            
        tiler.set_property("rows", 2)
        tiler.set_property("columns", 2)
        tiler.set_property("width", 1280)
        tiler.set_property("height", 720)
        
        # Create Video Converter
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write("Unable to create nvvidconv \n")
            return False
        
        # Create nvosd to draw on the converted RGBA buffer
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write("Unable to create nvosd \n")
            return False
        
        # Create sink
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write("Unable to create egl sink \n")
            return False
        
        sink.set_property("sync", False)
        
        # Add elements to pipeline
        self.pipeline.add(pgie)
        self.pipeline.add(tracker)
        self.pipeline.add(tiler)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        self.pipeline.add(sink)
        
        # Link elements
        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)
        
        # Create bus and add watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call)
        
        return True
    
    def bus_call(self, bus, message):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write(f"Error: {err}, {debug}\n")
            self.loop.quit()
        return True
    
    def start_pipeline(self):
        """Start the pipeline"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop = GObject.MainLoop()
            try:
                self.loop.run()
            except:
                pass
            
            # Cleanup
            self.pipeline.set_state(Gst.State.NULL)
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()

class MultiStreamManager:
    """Manages multiple video streams for crowd analysis"""
    
    def __init__(self):
        self.streams = {}
        self.pipeline = None
        
    def add_stream(self, stream_id, uri, location_name):
        """Add a new video stream"""
        self.streams[stream_id] = {
            'uri': uri,
            'location': location_name,
            'active': True,
            'crowd_data': {
                'density': 0,
                'count': 0,
                'movement_speed': 0,
                'alert_level': 'normal'
            }
        }
    
    def remove_stream(self, stream_id):
        """Remove a video stream"""
        if stream_id in self.streams:
            del self.streams[stream_id]
    
    def get_stream_uris(self):
        """Get list of all active stream URIs"""
        return [stream['uri'] for stream in self.streams.values() if stream['active']]
    
    def update_crowd_data(self, stream_id, crowd_data):
        """Update crowd analysis data for a stream"""
        if stream_id in self.streams:
            self.streams[stream_id]['crowd_data'].update(crowd_data)
    
    def get_all_crowd_data(self):
        """Get crowd data for all streams"""
        return {
            stream_id: {
                'location': stream['location'],
                'crowd_data': stream['crowd_data']
            }
            for stream_id, stream in self.streams.items()
            if stream['active']
        }