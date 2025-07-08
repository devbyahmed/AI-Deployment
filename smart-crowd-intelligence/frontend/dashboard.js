/**
 * Smart Crowd Intelligence Dashboard
 * Real-time crowd monitoring and analytics interface
 */

class SmartCrowdDashboard {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // Configuration
        this.config = {
            apiUrl: window.location.origin,
            websocketUrl: `ws://${window.location.host}/ws`,
            updateInterval: 1000,
            chartUpdateInterval: 5000
        };
        
        // Data storage
        this.streamData = new Map();
        this.alertsData = [];
        this.systemStatus = {};
        
        this.init();
    }
    
    /**
     * Initialize the dashboard
     */
    async init() {
        try {
            await this.setupEventListeners();
            await this.initializeCharts();
            await this.connectWebSocket();
            await this.loadInitialData();
            this.startPeriodicUpdates();
            this.updateSystemTime();
            
            console.log('Smart Crowd Intelligence Dashboard initialized successfully');
        } catch (error) {
            console.error('Dashboard initialization error:', error);
            this.showToast('Failed to initialize dashboard', 'error');
        }
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Window events
        window.addEventListener('beforeunload', () => this.cleanup());
        window.addEventListener('focus', () => this.handleWindowFocus());
        window.addEventListener('blur', () => this.handleWindowBlur());
        
        // Form submissions
        document.getElementById('streamForm')?.addEventListener('submit', (e) => this.handleStreamSubmit(e));
        
        // Button clicks
        document.querySelectorAll('[onclick]').forEach(element => {
            const onclick = element.getAttribute('onclick');
            element.removeAttribute('onclick');
            element.addEventListener('click', () => eval(onclick));
        });
        
        // Modal close events
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target.id);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }
    
    /**
     * Initialize Chart.js charts
     */
    async initializeCharts() {
        // Crowd trends chart
        const crowdTrendsCtx = document.getElementById('crowdTrendsChart');
        if (crowdTrendsCtx) {
            this.charts.crowdTrends = new Chart(crowdTrendsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Total Count',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: 'Density',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        fill: false,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Density'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    },
                    animation: {
                        duration: 750,
                        easing: 'easeInOutQuart'
                    }
                }
            });
        }
        
        // Movement chart
        const movementCtx = document.getElementById('movementChart');
        if (movementCtx) {
            this.charts.movement = new Chart(movementCtx, {
                type: 'radar',
                data: {
                    labels: ['North', 'NE', 'East', 'SE', 'South', 'SW', 'West', 'NW'],
                    datasets: [{
                        label: 'Movement Direction',
                        data: [0, 0, 0, 0, 0, 0, 0, 0],
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: '#9b59b6',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#9b59b6'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }
    
    /**
     * Connect to WebSocket server
     */
    async connectWebSocket() {
        try {
            this.websocket = new WebSocket(this.config.websocketUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Subscribe to all updates
                this.sendWebSocketMessage({
                    type: 'subscribe',
                    stream_id: 'all'
                });
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('WebSocket message parsing error:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showToast('Connection error', 'error');
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'stream_update':
                this.handleStreamUpdate(data.data);
                break;
            case 'alert':
                this.handleNewAlert(data.data);
                break;
            case 'system_status':
                this.handleSystemStatus(data.data);
                break;
            case 'alert_acknowledged':
                this.handleAlertAcknowledged(data.data);
                break;
            case 'alert_resolved':
                this.handleAlertResolved(data.data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    /**
     * Send WebSocket message
     */
    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        }
    }
    
    /**
     * Attempt to reconnect WebSocket
     */
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('Max reconnection attempts reached');
            this.showToast('Connection lost. Please refresh the page.', 'error');
        }
    }
    
    /**
     * Load initial data from API
     */
    async loadInitialData() {
        try {
            // Load system status
            const statusResponse = await fetch(`${this.config.apiUrl}/status`);
            if (statusResponse.ok) {
                this.systemStatus = await statusResponse.json();
                this.updateSystemMetrics();
            }
            
            // Load streams
            const streamsResponse = await fetch(`${this.config.apiUrl}/streams`);
            if (streamsResponse.ok) {
                const streamsData = await streamsResponse.json();
                this.updateStreamsDisplay(streamsData.streams);
            }
            
            // Load alerts
            const alertsResponse = await fetch(`${this.config.apiUrl}/alerts`);
            if (alertsResponse.ok) {
                const alertsData = await alertsResponse.json();
                this.updateAlertsDisplay(alertsData.alerts);
            }
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showToast('Failed to load initial data', 'error');
        }
    }
    
    /**
     * Handle stream updates
     */
    handleStreamUpdate(data) {
        this.streamData.set(data.stream_id, data);
        this.updateStreamCard(data.stream_id, data);
        this.updateAggregatedMetrics();
        this.updateCharts();
    }
    
    /**
     * Handle new alerts
     */
    handleNewAlert(alert) {
        this.alertsData.unshift(alert);
        this.updateAlertsDisplay(this.alertsData);
        this.showAlertNotification(alert);
        this.updateSystemMetrics();
    }
    
    /**
     * Handle system status updates
     */
    handleSystemStatus(status) {
        this.systemStatus = status;
        this.updateSystemMetrics();
    }
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connectionStatus');
        const text = document.getElementById('connectionText');
        
        if (indicator && text) {
            if (connected) {
                indicator.classList.remove('disconnected');
                text.textContent = 'Connected';
            } else {
                indicator.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }
    }
    
    /**
     * Update system metrics display
     */
    updateSystemMetrics() {
        // Calculate aggregated metrics
        let totalCount = 0;
        let avgDensity = 0;
        let activeStreams = this.streamData.size;
        let activeAlerts = this.alertsData.filter(a => !a.acknowledged).length;
        
        this.streamData.forEach(stream => {
            totalCount += stream.crowd_data?.count || 0;
            avgDensity += stream.crowd_data?.density || 0;
        });
        
        if (activeStreams > 0) {
            avgDensity = avgDensity / activeStreams;
        }
        
        // Update metric cards
        this.updateElement('currentCount', totalCount);
        this.updateElement('densityLevel', this.getDensityLabel(avgDensity));
        this.updateElement('densityValue', avgDensity.toFixed(1));
        this.updateElement('activeAlerts', activeAlerts);
        this.updateElement('activeStreams', activeStreams);
        this.updateElement('streamStatus', activeStreams > 0 ? 'Online' : 'Offline');
        
        // Update alert level
        const criticalAlerts = this.alertsData.filter(a => !a.acknowledged && a.level === 'CRITICAL').length;
        const highAlerts = this.alertsData.filter(a => !a.acknowledged && a.level === 'HIGH').length;
        
        let alertLevel = 'Normal';
        if (criticalAlerts > 0) alertLevel = 'Critical';
        else if (highAlerts > 0) alertLevel = 'High';
        else if (activeAlerts > 0) alertLevel = 'Medium';
        
        this.updateElement('alertLevel', alertLevel);
    }
    
    /**
     * Update charts with new data
     */
    updateCharts() {
        this.updateCrowdTrendsChart();
        this.updateMovementChart();
        this.updateDensityHeatmap();
    }
    
    /**
     * Update crowd trends chart
     */
    updateCrowdTrendsChart() {
        if (!this.charts.crowdTrends) return;
        
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        
        // Calculate totals
        let totalCount = 0;
        let avgDensity = 0;
        let streamCount = 0;
        
        this.streamData.forEach(stream => {
            totalCount += stream.crowd_data?.count || 0;
            avgDensity += stream.crowd_data?.density || 0;
            streamCount++;
        });
        
        if (streamCount > 0) {
            avgDensity = avgDensity / streamCount;
        }
        
        // Update chart data
        const chart = this.charts.crowdTrends;
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(totalCount);
        chart.data.datasets[1].data.push(avgDensity);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        chart.update('none');
    }
    
    /**
     * Update movement chart
     */
    updateMovementChart() {
        if (!this.charts.movement) return;
        
        // Aggregate movement data from all streams
        const movementData = [0, 0, 0, 0, 0, 0, 0, 0]; // 8 directions
        let totalMovement = 0;
        
        this.streamData.forEach(stream => {
            const movement = stream.crowd_data?.movement;
            if (movement && movement.directions) {
                movement.directions.forEach((value, index) => {
                    movementData[index] += value;
                    totalMovement += value;
                });
            }
        });
        
        // Normalize to percentages
        if (totalMovement > 0) {
            for (let i = 0; i < movementData.length; i++) {
                movementData[i] = (movementData[i] / totalMovement) * 100;
            }
        }
        
        this.charts.movement.data.datasets[0].data = movementData;
        this.charts.movement.update('none');
        
        // Update movement metrics
        let avgSpeed = 0;
        let dominantDirection = 'None';
        let congestionLevel = 'Low';
        
        this.streamData.forEach(stream => {
            const movement = stream.crowd_data?.movement;
            if (movement) {
                avgSpeed += movement.avg_speed || 0;
                if (movement.congestion_level) {
                    congestionLevel = movement.congestion_level;
                }
            }
        });
        
        if (this.streamData.size > 0) {
            avgSpeed = avgSpeed / this.streamData.size;
        }
        
        // Find dominant direction
        const maxIndex = movementData.indexOf(Math.max(...movementData));
        const directions = ['North', 'NE', 'East', 'SE', 'South', 'SW', 'West', 'NW'];
        if (movementData[maxIndex] > 20) {
            dominantDirection = directions[maxIndex];
        }
        
        this.updateElement('avgSpeed', `${avgSpeed.toFixed(1)} m/s`);
        this.updateElement('flowDirection', dominantDirection);
        this.updateElement('congestionLevel', congestionLevel);
    }
    
    /**
     * Update density heatmap
     */
    updateDensityHeatmap() {
        const container = document.getElementById('densityHeatmap');
        if (!container) return;
        
        // Simple visualization of density hotspots
        container.innerHTML = '';
        
        this.streamData.forEach((stream, streamId) => {
            const density = stream.crowd_data?.density || 0;
            const hotspot = document.createElement('div');
            hotspot.style.position = 'absolute';
            hotspot.style.width = '20px';
            hotspot.style.height = '20px';
            hotspot.style.borderRadius = '50%';
            hotspot.style.left = Math.random() * 80 + '%';
            hotspot.style.top = Math.random() * 80 + '%';
            
            if (density > 25) {
                hotspot.style.backgroundColor = '#e74c3c';
            } else if (density > 15) {
                hotspot.style.backgroundColor = '#f39c12';
            } else if (density > 5) {
                hotspot.style.backgroundColor = '#f1c40f';
            } else {
                hotspot.style.backgroundColor = '#2ecc71';
            }
            
            hotspot.title = `${streamId}: ${density.toFixed(1)} people/m²`;
            container.appendChild(hotspot);
        });
    }
    
    /**
     * Update streams display
     */
    updateStreamsDisplay(streams) {
        const grid = document.getElementById('streamsGrid');
        if (!grid) return;
        
        grid.innerHTML = '';
        
        streams.forEach(stream => {
            const card = this.createStreamCard(stream);
            grid.appendChild(card);
        });
    }
    
    /**
     * Create stream card element
     */
    createStreamCard(stream) {
        const card = document.createElement('div');
        card.className = `stream-card ${stream.active ? 'active' : ''}`;
        card.dataset.streamId = stream.stream_id;
        
        card.innerHTML = `
            <div class="stream-preview">
                <i class="fas fa-video"></i>
            </div>
            <div class="stream-info">
                <h4>${stream.stream_id}</h4>
                <p>${stream.location}</p>
                <small>Count: ${stream.crowd_data?.count || 0} | Density: ${stream.crowd_data?.density?.toFixed(1) || 0}</small>
            </div>
        `;
        
        card.addEventListener('click', () => {
            this.selectStream(stream.stream_id);
        });
        
        return card;
    }
    
    /**
     * Update specific stream card
     */
    updateStreamCard(streamId, data) {
        const card = document.querySelector(`[data-stream-id="${streamId}"]`);
        if (!card) return;
        
        const info = card.querySelector('.stream-info small');
        if (info) {
            info.textContent = `Count: ${data.crowd_data?.count || 0} | Density: ${data.crowd_data?.density?.toFixed(1) || 0}`;
        }
        
        // Update behavior indicator
        this.updateBehaviorAnalysis(data.crowd_data);
    }
    
    /**
     * Update behavior analysis display
     */
    updateBehaviorAnalysis(crowdData) {
        if (!crowdData || !crowdData.behavior) return;
        
        const indicator = document.getElementById('behaviorIndicator');
        const type = document.getElementById('behaviorType');
        const confidence = document.getElementById('behaviorConfidence');
        const riskLevel = document.getElementById('riskLevel');
        const riskFactors = document.getElementById('riskFactors');
        
        if (indicator && type && confidence) {
            const behavior = crowdData.behavior;
            
            // Update behavior type and confidence
            type.textContent = this.formatBehaviorType(behavior.type);
            confidence.textContent = `${(behavior.confidence * 100).toFixed(0)}% confidence`;
            
            // Update indicator color
            indicator.className = 'behavior-indicator';
            if (behavior.type.includes('panic') || behavior.type.includes('anomalous')) {
                indicator.classList.add('danger');
            } else if (behavior.type.includes('congestion') || behavior.type.includes('bottleneck')) {
                indicator.classList.add('warning');
            }
            
            // Update risk assessment
            if (riskLevel && riskFactors) {
                const risk = crowdData.risk_assessment || { level: 'low', factors: [] };
                riskLevel.textContent = `${risk.level.charAt(0).toUpperCase() + risk.level.slice(1)} Risk`;
                riskLevel.className = `risk-level ${risk.level}`;
                riskFactors.textContent = risk.factors.join(', ') || 'No significant risk factors detected';
            }
        }
    }
    
    /**
     * Update alerts display
     */
    updateAlertsDisplay(alerts) {
        const container = document.getElementById('alertsList');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Filter alerts based on current filter
        const filter = document.getElementById('alertFilter')?.value || 'all';
        const filteredAlerts = filter === 'all' ? alerts : alerts.filter(a => a.level.toLowerCase() === filter);
        
        filteredAlerts.slice(0, 10).forEach(alert => {
            const alertElement = this.createAlertElement(alert);
            container.appendChild(alertElement);
        });
    }
    
    /**
     * Create alert element
     */
    createAlertElement(alert) {
        const element = document.createElement('div');
        element.className = `alert-item ${alert.level.toLowerCase()}`;
        element.dataset.alertId = alert.id;
        
        const timeAgo = this.getTimeAgo(new Date(alert.timestamp));
        
        element.innerHTML = `
            <div class="alert-header">
                <span class="alert-level ${alert.level.toLowerCase()}">${alert.level}</span>
                <span class="alert-time">${timeAgo}</span>
            </div>
            <div class="alert-message">${alert.message}</div>
            ${alert.stream_id ? `<div class="alert-stream">Stream: ${alert.stream_id}</div>` : ''}
            ${!alert.acknowledged ? '<button onclick="acknowledgeAlert(\'' + alert.id + '\')">Acknowledge</button>' : ''}
        `;
        
        return element;
    }
    
    /**
     * Show alert notification
     */
    showAlertNotification(alert) {
        // Visual notification
        this.showToast(`${alert.level}: ${alert.message}`, alert.level.toLowerCase());
        
        // Audio notification (if enabled)
        if (this.getSetting('enableSounds', true)) {
            this.playAlertSound(alert.level);
        }
        
        // Browser notification (if permission granted)
        if (Notification.permission === 'granted') {
            new Notification(`Crowd Alert - ${alert.level}`, {
                body: alert.message,
                icon: '/favicon.ico',
                tag: alert.id
            });
        }
    }
    
    /**
     * Start periodic updates
     */
    startPeriodicUpdates() {
        // Update system time
        setInterval(() => {
            this.updateSystemTime();
        }, 1000);
        
        // Periodic data refresh (if WebSocket is not connected)
        setInterval(() => {
            if (!this.isConnected) {
                this.loadInitialData();
            }
        }, 30000);
        
        // Chart updates
        setInterval(() => {
            if (this.isConnected) {
                this.updateCharts();
            }
        }, this.config.chartUpdateInterval);
    }
    
    /**
     * Update system time display
     */
    updateSystemTime() {
        const timeElement = document.getElementById('systemTime');
        if (timeElement) {
            timeElement.textContent = new Date().toLocaleString();
        }
    }
    
    /**
     * Utility functions
     */
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    getDensityLabel(density) {
        if (density > 25) return 'Critical';
        if (density > 15) return 'High';
        if (density > 5) return 'Medium';
        return 'Low';
    }
    
    formatBehaviorType(type) {
        return type.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return 'Just now';
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
    
    playAlertSound(level) {
        try {
            const audio = new Audio();
            switch(level) {
                case 'CRITICAL':
                    audio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEcB...';
                    break;
                case 'HIGH':
                    audio.src = 'data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU...';
                    break;
                default:
                    return;
            }
            audio.play().catch(() => {}); // Ignore errors
        } catch (error) {
            // Audio not supported or failed
        }
    }
    
    getSetting(key, defaultValue) {
        try {
            const value = localStorage.getItem(`sci_${key}`);
            return value !== null ? JSON.parse(value) : defaultValue;
        } catch {
            return defaultValue;
        }
    }
    
    setSetting(key, value) {
        try {
            localStorage.setItem(`sci_${key}`, JSON.stringify(value));
        } catch (error) {
            console.error('Failed to save setting:', error);
        }
    }
    
    /**
     * Window focus/blur handlers
     */
    handleWindowFocus() {
        // Resume real-time updates
        if (this.websocket && this.websocket.readyState === WebSocket.CLOSED) {
            this.connectWebSocket();
        }
    }
    
    handleWindowBlur() {
        // Can reduce update frequency when window is not focused
    }
    
    /**
     * Keyboard shortcuts
     */
    handleKeyboard(event) {
        // Ctrl/Cmd + R: Refresh data
        if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
            event.preventDefault();
            this.loadInitialData();
        }
        
        // Escape: Close modals
        if (event.key === 'Escape') {
            document.querySelectorAll('.modal').forEach(modal => {
                if (modal.style.display === 'block') {
                    this.closeModal(modal.id);
                }
            });
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        // Clean up charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Global functions for HTML event handlers
let dashboard;

function addStream() {
    document.getElementById('streamModal').style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function openSettings() {
    document.getElementById('settingsModal').style.display = 'block';
}

function saveSettings() {
    const densityThreshold = document.getElementById('densityThreshold').value;
    const crowdThreshold = document.getElementById('crowdThreshold').value;
    const enableSounds = document.getElementById('enableSounds').checked;
    const autoRefresh = document.getElementById('autoRefresh').checked;
    
    dashboard.setSetting('densityThreshold', densityThreshold);
    dashboard.setSetting('crowdThreshold', crowdThreshold);
    dashboard.setSetting('enableSounds', enableSounds);
    dashboard.setSetting('autoRefresh', autoRefresh);
    
    dashboard.showToast('Settings saved successfully', 'success');
    closeModal('settingsModal');
}

async function submitStream(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const streamData = {
        stream_id: formData.get('streamId'),
        uri: formData.get('streamUri'),
        location: formData.get('streamLocation')
    };
    
    try {
        const response = await fetch('/streams', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(streamData)
        });
        
        if (response.ok) {
            dashboard.showToast('Stream added successfully', 'success');
            closeModal('streamModal');
            dashboard.loadInitialData();
        } else {
            const error = await response.json();
            dashboard.showToast(error.detail || 'Failed to add stream', 'error');
        }
    } catch (error) {
        dashboard.showToast('Network error', 'error');
    }
}

async function acknowledgeAlert(alertId) {
    try {
        const response = await fetch(`/alerts/${alertId}/acknowledge`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user: 'dashboard_user'
            })
        });
        
        if (response.ok) {
            dashboard.showToast('Alert acknowledged', 'success');
            // Update will come via WebSocket
        } else {
            dashboard.showToast('Failed to acknowledge alert', 'error');
        }
    } catch (error) {
        dashboard.showToast('Network error', 'error');
    }
}

function filterAlerts() {
    const filter = document.getElementById('alertFilter').value;
    dashboard.updateAlertsDisplay(dashboard.alertsData);
}

function updateCharts() {
    const timeRange = document.getElementById('timeRange').value;
    // Implementation for historical data loading based on time range
    dashboard.showToast(`Loading data for ${timeRange}`, 'info');
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new SmartCrowdDashboard();
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SmartCrowdDashboard;
}