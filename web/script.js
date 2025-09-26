/**
 * Error Analysis Dashboard JavaScript
 * Handles data loading, UI interactions, and real-time updates
 */

class ErrorDashboard {
    constructor() {
        this.data = null;
        this.currentTab = 'overview';
        this.currentPage = 1;
        this.itemsPerPage = 25;
        this.filters = {
            severity: '',
            service: '',
            search: ''
        };
        this.sortBy = 'severity';
        this.autoRefresh = true;
        this.refreshInterval = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadData();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadData();
        });

        // Export button
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportData();
        });

        // Search input
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.filters.search = e.target.value;
            this.filterAndDisplayIssues();
        });

        // Filter controls
        document.getElementById('severity-filter').addEventListener('change', (e) => {
            this.filters.severity = e.target.value;
            this.filterAndDisplayIssues();
        });

        document.getElementById('service-filter').addEventListener('change', (e) => {
            this.filters.service = e.target.value;
            this.filterAndDisplayIssues();
        });

        document.getElementById('sort-filter').addEventListener('change', (e) => {
            this.sortBy = e.target.value;
            this.filterAndDisplayIssues();
        });

        // Pagination
        document.getElementById('prev-page').addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.filterAndDisplayIssues();
            }
        });

        document.getElementById('next-page').addEventListener('click', () => {
            const totalPages = Math.ceil(this.getFilteredIssues().length / this.itemsPerPage);
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.filterAndDisplayIssues();
            }
        });

        // Analysis controls
        document.getElementById('run-analysis-btn').addEventListener('click', () => {
            this.runAnalysis();
        });

        document.getElementById('view-logs-btn').addEventListener('click', () => {
            this.viewLogs();
        });

        // Settings
        document.getElementById('auto-refresh').addEventListener('change', (e) => {
            this.autoRefresh = e.target.checked;
            if (this.autoRefresh) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });

        document.getElementById('items-per-page').addEventListener('change', (e) => {
            this.itemsPerPage = parseInt(e.target.value);
            this.currentPage = 1;
            this.filterAndDisplayIssues();
        });

        // Modal controls
        document.getElementById('modal-close').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('modal-close-btn').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('modal-action-btn').addEventListener('click', () => {
            this.takeAction();
        });

        // Close modal on outside click
        document.getElementById('issue-modal').addEventListener('click', (e) => {
            if (e.target.id === 'issue-modal') {
                this.closeModal();
            }
        });
    }

    async loadData() {
        this.showLoading();
        
        try {
            const response = await fetch('/reports/latest.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.data = await response.json();
            this.updateUI();
            this.updateStatus('connected');
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data. Please check the connection and try again.');
            this.updateStatus('error');
        } finally {
            this.hideLoading();
        }
    }

    updateUI() {
        if (!this.data) return;

        this.updateSummaryCards();
        this.updateCharts();
        this.updateRecentIssues();
        this.updateAnalysisDetails();
        this.populateServiceFilter();
        this.filterAndDisplayIssues();
    }

    updateSummaryCards() {
        const summary = this.data.summary;
        
        document.getElementById('total-issues').textContent = summary.total_issues || 0;
        document.getElementById('level-1-issues').textContent = summary.level_1_issues || 0;
        document.getElementById('level-2-issues').textContent = summary.level_2_issues || 0;
        document.getElementById('level-3-issues').textContent = summary.level_3_issues || 0;
        document.getElementById('action-needed').textContent = summary.human_action_needed || 0;
        document.getElementById('total-cost').textContent = `$${(summary.total_cost || 0).toFixed(4)}`;
        
        // Update last updated time
        const lastUpdated = new Date(this.data.analysis_date);
        document.getElementById('last-updated-time').textContent = lastUpdated.toLocaleString();
    }

    updateCharts() {
        this.updateSeverityChart();
        this.updateCostChart();
    }

    updateSeverityChart() {
        const ctx = document.getElementById('severity-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.severityChart) {
            this.severityChart.destroy();
        }

        const summary = this.data.summary;
        this.severityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Level 1', 'Level 2', 'Level 3'],
                datasets: [{
                    data: [
                        summary.level_1_issues || 0,
                        summary.level_2_issues || 0,
                        summary.level_3_issues || 0
                    ],
                    backgroundColor: ['#1976d2', '#f57c00', '#d32f2f'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    updateCostChart() {
        const ctx = document.getElementById('cost-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.costChart) {
            this.costChart.destroy();
        }

        const issues = this.data.issues || [];
        const modelCosts = {};
        
        issues.forEach(issue => {
            const model = issue.analysis_model || 'unknown';
            const cost = issue.estimated_cost || 0;
            modelCosts[model] = (modelCosts[model] || 0) + cost;
        });

        const labels = Object.keys(modelCosts);
        const data = Object.values(modelCosts);

        this.costChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cost ($)',
                    data: data,
                    backgroundColor: '#667eea',
                    borderColor: '#5a6fd8',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(4);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    updateRecentIssues() {
        const issues = this.data.issues || [];
        const recentIssues = issues.slice(0, 5);
        
        const container = document.getElementById('recent-issues-list');
        
        if (recentIssues.length === 0) {
            container.innerHTML = '<div class="loading">No recent issues found</div>';
            return;
        }

        container.innerHTML = recentIssues.map(issue => this.createIssueCard(issue)).join('');
    }

    createIssueCard(issue) {
        const severityClass = `severity-${issue.severity.toLowerCase()}`;
        const severityText = issue.severity.replace('LEVEL_', 'Level ');
        
        return `
            <div class="issue-item" onclick="dashboard.showIssueDetails('${issue.issue_id}')">
                <div class="issue-header">
                    <div class="issue-id">${issue.issue_id}</div>
                    <div class="severity-badge ${severityClass}">${severityText}</div>
                </div>
                <div class="issue-content">
                    <div class="issue-message">${this.truncateText(issue.error_message, 100)}</div>
                    <div class="issue-details">
                        <div class="detail-item">
                            <div class="detail-label">Service</div>
                            <div class="detail-value">${issue.service || 'Unknown'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Count</div>
                            <div class="detail-value">${issue.count || 0}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Cost</div>
                            <div class="detail-value">$${(issue.estimated_cost || 0).toFixed(4)}</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    updateAnalysisDetails() {
        if (!this.data) return;

        document.getElementById('analysis-date').textContent = 
            new Date(this.data.analysis_date).toLocaleString();
        
        const dateRange = this.data.date_range;
        document.getElementById('analysis-range').textContent = 
            `${new Date(dateRange.start).toLocaleDateString()} - ${new Date(dateRange.end).toLocaleDateString()}`;
        
        document.getElementById('analysis-site').textContent = this.data.site || 'Unknown';
        
        // Calculate processing time (mock for now)
        document.getElementById('processing-time').textContent = '2.3 seconds';
    }

    populateServiceFilter() {
        const serviceFilter = document.getElementById('service-filter');
        const issues = this.data.issues || [];
        const services = [...new Set(issues.map(issue => issue.service).filter(Boolean))];
        
        // Clear existing options except "All Services"
        serviceFilter.innerHTML = '<option value="">All Services</option>';
        
        services.forEach(service => {
            const option = document.createElement('option');
            option.value = service;
            option.textContent = service;
            serviceFilter.appendChild(option);
        });
    }

    filterAndDisplayIssues() {
        const filteredIssues = this.getFilteredIssues();
        const paginatedIssues = this.getPaginatedIssues(filteredIssues);
        
        this.displayIssues(paginatedIssues);
        this.updatePagination(filteredIssues.length);
    }

    getFilteredIssues() {
        if (!this.data || !this.data.issues) return [];
        
        return this.data.issues.filter(issue => {
            // Severity filter
            if (this.filters.severity && issue.severity !== this.filters.severity) {
                return false;
            }
            
            // Service filter
            if (this.filters.service && issue.service !== this.filters.service) {
                return false;
            }
            
            // Search filter
            if (this.filters.search) {
                const searchTerm = this.filters.search.toLowerCase();
                const searchableText = [
                    issue.error_message,
                    issue.service,
                    issue.error_type,
                    issue.issue_id
                ].join(' ').toLowerCase();
                
                if (!searchableText.includes(searchTerm)) {
                    return false;
                }
            }
            
            return true;
        }).sort((a, b) => {
            switch (this.sortBy) {
                case 'severity':
                    const severityOrder = { 'LEVEL_3': 3, 'LEVEL_2': 2, 'LEVEL_1': 1 };
                    return (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0);
                case 'timestamp':
                    return new Date(b.timestamp) - new Date(a.timestamp);
                case 'cost':
                    return (b.estimated_cost || 0) - (a.estimated_cost || 0);
                default:
                    return 0;
            }
        });
    }

    getPaginatedIssues(issues) {
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        return issues.slice(startIndex, endIndex);
    }

    displayIssues(issues) {
        const container = document.getElementById('issues-container');
        
        if (issues.length === 0) {
            container.innerHTML = '<div class="loading">No issues found matching the current filters</div>';
            return;
        }

        container.innerHTML = issues.map(issue => this.createIssueCard(issue)).join('');
    }

    updatePagination(totalItems) {
        const totalPages = Math.ceil(totalItems / this.itemsPerPage);
        const pageInfo = document.getElementById('page-info');
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        
        pageInfo.textContent = `Page ${this.currentPage} of ${totalPages}`;
        prevBtn.disabled = this.currentPage <= 1;
        nextBtn.disabled = this.currentPage >= totalPages;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load tab-specific data if needed
        if (tabName === 'issues') {
            this.filterAndDisplayIssues();
        }
    }

    showIssueDetails(issueId) {
        const issue = this.data.issues.find(i => i.issue_id === issueId);
        if (!issue) return;
        
        const modal = document.getElementById('issue-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');
        
        modalTitle.textContent = `Issue Details: ${issue.issue_id}`;
        
        modalBody.innerHTML = this.createIssueDetailsHTML(issue);
        
        modal.classList.add('show');
        this.currentIssue = issue;
    }

    createIssueDetailsHTML(issue) {
        const severityClass = `severity-${issue.severity.toLowerCase()}`;
        const severityText = issue.severity.replace('LEVEL_', 'Level ');
        
        return `
            <div class="issue-details-full">
                <div class="issue-header">
                    <div class="issue-id">${issue.issue_id}</div>
                    <div class="severity-badge ${severityClass}">${severityText}</div>
                </div>
                
                <div class="issue-content">
                    <h4>Error Message</h4>
                    <p class="issue-message">${issue.error_message}</p>
                    
                    <h4>Error Type</h4>
                    <p>${issue.error_type || 'Unknown'}</p>
                    
                    <h4>Service Information</h4>
                    <div class="service-info">
                        <div class="detail-item">
                            <div class="detail-label">Service</div>
                            <div class="detail-value">${issue.service || 'Unknown'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Count</div>
                            <div class="detail-value">${issue.count || 0}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">First Occurrence</div>
                            <div class="detail-value">${new Date(issue.timestamp).toLocaleString()}</div>
                        </div>
                    </div>
                    
                    <h4>Analysis Results</h4>
                    <div class="analysis-results">
                        <div class="detail-item">
                            <div class="detail-label">Confidence Score</div>
                            <div class="detail-value">${(issue.confidence_score || 0).toFixed(2)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Analysis Model</div>
                            <div class="detail-value">${issue.analysis_model || 'Unknown'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Estimated Cost</div>
                            <div class="detail-value">$${(issue.estimated_cost || 0).toFixed(4)}</div>
                        </div>
                    </div>
                    
                    ${issue.remediation_plan && issue.remediation_plan.human_action_needed ? `
                        <div class="action-required">
                            <h4><i class="fas fa-exclamation-triangle"></i> Human Action Required</h4>
                            <p><strong>Root Cause:</strong> ${issue.remediation_plan.root_cause || 'Not specified'}</p>
                            <p><strong>Urgency:</strong> ${issue.remediation_plan.urgency || 'MEDIUM'}</p>
                            <p><strong>Damaged Modules:</strong> ${(issue.remediation_plan.damaged_modules || []).join(', ') || 'None identified'}</p>
                            <ul class="action-guidelines">
                                ${(issue.remediation_plan.action_guidelines || []).map(guideline => 
                                    `<li>${guideline}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${issue.rds_integrity_check ? `
                        <h4>RDS Data Integrity Check</h4>
                        <div class="integrity-check">
                            <div class="detail-item">
                                <div class="detail-label">Status</div>
                                <div class="detail-value">${issue.rds_integrity_check.data_integrity_status || 'UNKNOWN'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Checks Performed</div>
                                <div class="detail-value">${(issue.rds_integrity_check.checks_performed || []).join(', ') || 'None'}</div>
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    closeModal() {
        const modal = document.getElementById('issue-modal');
        modal.classList.remove('show');
        this.currentIssue = null;
    }

    takeAction() {
        if (!this.currentIssue) return;
        
        // Implement action taking logic
        alert(`Taking action for issue: ${this.currentIssue.issue_id}`);
        this.closeModal();
    }

    exportData() {
        if (!this.data) return;
        
        const exportFormat = document.getElementById('export-format').value;
        const filename = `error_analysis_${new Date().toISOString().split('T')[0]}.${exportFormat}`;
        
        let content, mimeType;
        
        switch (exportFormat) {
            case 'json':
                content = JSON.stringify(this.data, null, 2);
                mimeType = 'application/json';
                break;
            case 'csv':
                content = this.convertToCSV(this.data.issues || []);
                mimeType = 'text/csv';
                break;
            default:
                alert('Export format not supported yet');
                return;
        }
        
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    convertToCSV(issues) {
        if (issues.length === 0) return '';
        
        const headers = [
            'Issue ID', 'Severity', 'Error Message', 'Service', 'Count', 
            'Error Type', 'Confidence Score', 'Analysis Model', 'Estimated Cost',
            'Human Action Needed', 'Root Cause', 'Urgency'
        ];
        
        const rows = issues.map(issue => [
            issue.issue_id || '',
            issue.severity || '',
            `"${(issue.error_message || '').replace(/"/g, '""')}"`,
            issue.service || '',
            issue.count || 0,
            issue.error_type || '',
            issue.confidence_score || 0,
            issue.analysis_model || '',
            issue.estimated_cost || 0,
            issue.remediation_plan?.human_action_needed || false,
            `"${(issue.remediation_plan?.root_cause || '').replace(/"/g, '""')}"`,
            issue.remediation_plan?.urgency || ''
        ]);
        
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    runAnalysis() {
        this.showLoading();
        // Implement analysis running logic
        setTimeout(() => {
            this.hideLoading();
            alert('Analysis completed! Data has been refreshed.');
            this.loadData();
        }, 2000);
    }

    viewLogs() {
        // Implement log viewing logic
        alert('Log viewer not implemented yet');
    }

    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            if (this.autoRefresh) {
                this.loadData();
            }
        }, 30000); // 30 seconds
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    updateStatus(status) {
        const indicator = document.getElementById('status-indicator');
        const icon = indicator.querySelector('i');
        const text = indicator.querySelector('span');
        
        indicator.className = `status-indicator ${status}`;
        
        switch (status) {
            case 'connected':
                icon.className = 'fas fa-circle';
                text.textContent = 'Connected';
                break;
            case 'error':
                icon.className = 'fas fa-circle';
                text.textContent = 'Error';
                break;
            case 'loading':
                icon.className = 'fas fa-spinner fa-spin';
                text.textContent = 'Loading...';
                break;
        }
    }

    showLoading() {
        document.getElementById('loading-overlay').classList.add('show');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.remove('show');
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        // Insert at the top of the main content
        const main = document.querySelector('.dashboard-main');
        main.insertBefore(errorDiv, main.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ErrorDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        if (document.hidden) {
            window.dashboard.stopAutoRefresh();
        } else {
            window.dashboard.startAutoRefresh();
        }
    }
});

// Handle window beforeunload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.stopAutoRefresh();
    }
});
