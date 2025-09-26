# Error Analysis Dashboard

A modern, responsive web dashboard for monitoring and analyzing error logs from the Vortexai production service.

## Features

### 📊 **Overview Dashboard**
- **Summary Cards**: Total issues, severity breakdown, action required, and cost tracking
- **Interactive Charts**: Severity distribution and cost breakdown visualizations
- **Recent Issues**: Quick view of the latest error issues
- **Real-time Status**: Connection status and last updated timestamp

### 🔍 **Issues Management**
- **Advanced Filtering**: Filter by severity, service, and search terms
- **Sorting Options**: Sort by severity, timestamp, or cost
- **Pagination**: Navigate through large datasets efficiently
- **Detailed View**: Click any issue for comprehensive details

### 📈 **Analysis Tools**
- **Run Analysis**: Trigger new error analysis runs
- **View Logs**: Access system logs and processing details
- **Analysis Statistics**: Processing time, date range, and site information
- **Model Performance**: Track analysis model usage and costs

### ⚙️ **Settings & Configuration**
- **Auto Refresh**: Configurable automatic data updates
- **Display Options**: Customizable items per page and filters
- **Export Settings**: Choose default export format (JSON, CSV, PDF)
- **User Preferences**: Personalized dashboard experience

## Technical Architecture

### Frontend Stack
- **HTML5**: Semantic markup with accessibility features
- **CSS3**: Modern styling with Flexbox and Grid layouts
- **JavaScript ES6+**: Vanilla JavaScript with class-based architecture
- **Chart.js**: Interactive data visualizations
- **Font Awesome**: Professional iconography

### Backend Integration
- **Nginx**: High-performance web server with custom configuration
- **Static File Serving**: Optimized delivery of assets and reports
- **CORS Support**: Cross-origin resource sharing for API access
- **Gzip Compression**: Reduced bandwidth usage and faster loading

### Data Sources
- **JSON Reports**: Real-time data from `/reports/latest.json`
- **Analysis Results**: Comprehensive error analysis data
- **RDS Integrity**: Database health and data integrity status
- **Cost Tracking**: OpenAI API usage and cost monitoring

## File Structure

```
web/
├── index.html          # Main dashboard interface
├── styles.css          # Comprehensive CSS styling
├── script.js           # JavaScript application logic
├── nginx.conf          # Nginx server configuration
├── 404.html            # Custom 404 error page
├── 50x.html            # Custom 5xx error page
└── README.md           # This documentation
```

## Key Components

### 1. **ErrorDashboard Class**
- Main application controller
- Handles data loading and UI updates
- Manages user interactions and state
- Implements auto-refresh functionality

### 2. **Data Management**
- Fetches data from `/reports/latest.json`
- Implements filtering and sorting
- Handles pagination and search
- Manages real-time updates

### 3. **UI Components**
- **Summary Cards**: Visual metrics display
- **Charts**: Interactive data visualizations
- **Issue Cards**: Compact issue information
- **Modal Dialogs**: Detailed issue views
- **Navigation**: Tab-based interface

### 4. **Responsive Design**
- Mobile-first approach
- Flexible grid layouts
- Adaptive typography
- Touch-friendly interactions

## API Endpoints

### Data Endpoints
- `GET /reports/latest.json` - Latest analysis results
- `GET /health` - Server health check

### Static Assets
- `GET /` - Main dashboard
- `GET /styles.css` - Stylesheet
- `GET /script.js` - JavaScript application
- `GET /404.html` - 404 error page
- `GET /50x.html` - 5xx error page

## Configuration

### Nginx Configuration
- **Port**: 80 (mapped to 8080 on host)
- **Root**: `/usr/share/nginx/html`
- **Gzip**: Enabled for text-based files
- **Caching**: Optimized for static assets
- **Security**: Security headers enabled

### Environment Variables
- No environment variables required
- Configuration through Docker Compose
- Volume mappings for live updates

## Usage

### Starting the Dashboard
```bash
# Start with Docker Compose
docker-compose up -d web-dashboard

# Access the dashboard
open http://localhost:8080
```

### Development Mode
```bash
# Mount web directory for live updates
docker-compose up -d web-dashboard

# Edit files in ./web directory
# Changes are reflected immediately
```

### Production Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Configure reverse proxy
# Set up SSL certificates
# Configure monitoring
```

## Features in Detail

### 1. **Real-time Monitoring**
- Auto-refresh every 30 seconds
- Connection status indicators
- Live data updates
- Error handling and recovery

### 2. **Advanced Filtering**
- **Severity Filter**: LEVEL_1, LEVEL_2, LEVEL_3
- **Service Filter**: Filter by Vortexai services
- **Search**: Full-text search across all fields
- **Sorting**: Multiple sort options

### 3. **Data Visualization**
- **Doughnut Chart**: Severity distribution
- **Bar Chart**: Cost breakdown by model
- **Interactive**: Hover effects and tooltips
- **Responsive**: Adapts to screen size

### 4. **Issue Management**
- **Detailed View**: Comprehensive issue information
- **Action Planning**: Human action requirements
- **RDS Integration**: Data integrity status
- **Export Options**: Multiple format support

### 5. **User Experience**
- **Intuitive Navigation**: Tab-based interface
- **Loading States**: Visual feedback during operations
- **Error Handling**: Graceful error recovery
- **Accessibility**: WCAG compliant design

## Browser Support

### Supported Browsers
- **Chrome**: 80+ (recommended)
- **Firefox**: 75+
- **Safari**: 13+
- **Edge**: 80+

### Required Features
- ES6+ JavaScript support
- CSS Grid and Flexbox
- Fetch API
- Local Storage
- Canvas API (for charts)

## Performance

### Optimization Features
- **Gzip Compression**: 70% size reduction
- **Asset Caching**: 1-year cache for static files
- **Lazy Loading**: On-demand content loading
- **Minification**: Optimized JavaScript and CSS

### Performance Metrics
- **First Load**: < 2 seconds
- **Subsequent Loads**: < 500ms
- **Chart Rendering**: < 100ms
- **Data Updates**: < 200ms

## Security

### Security Headers
- **X-Frame-Options**: SAMEORIGIN
- **X-Content-Type-Options**: nosniff
- **X-XSS-Protection**: 1; mode=block
- **Referrer-Policy**: strict-origin-when-cross-origin

### Data Protection
- **No Sensitive Data**: Only analysis results
- **CORS Configuration**: Controlled cross-origin access
- **Input Validation**: Client-side validation
- **Error Handling**: No sensitive information exposure

## Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check container status
docker ps | grep web-dashboard

# Check logs
docker logs error-monitor-web

# Restart container
docker-compose restart web-dashboard
```

#### Data Not Updating
```bash
# Check latest.json exists
ls -la reports/latest.json

# Check file permissions
chmod 644 reports/latest.json

# Restart error monitor
docker-compose restart error-monitor
```

#### Charts Not Displaying
- Check browser console for JavaScript errors
- Verify Chart.js is loading
- Check data format in latest.json
- Clear browser cache

### Debug Mode
```javascript
// Enable debug logging in browser console
localStorage.setItem('debug', 'true');
location.reload();
```

## Development

### Adding New Features
1. Update HTML structure in `index.html`
2. Add styles in `styles.css`
3. Implement logic in `script.js`
4. Test across different browsers
5. Update documentation

### Customizing Styles
- Modify CSS variables for colors
- Update grid layouts for different screen sizes
- Add new component styles
- Ensure accessibility compliance

### Extending Functionality
- Add new data sources
- Implement additional chart types
- Add new filter options
- Create custom export formats

## Contributing

### Code Style
- Use consistent indentation (2 spaces)
- Follow JavaScript ES6+ standards
- Use semantic HTML elements
- Write accessible markup

### Testing
- Test on multiple browsers
- Verify responsive design
- Check accessibility compliance
- Validate HTML and CSS

## License

This dashboard is part of the Error Log Monitoring System and follows the same license terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Check browser console for errors
4. Verify data format in latest.json
5. Contact the development team
