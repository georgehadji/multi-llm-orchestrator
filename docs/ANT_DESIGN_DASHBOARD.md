# Ant Design Dashboard v3.0

Modern, professional dashboard using Ant Design System.

## 🎨 Features

- **Modern UI**: Clean, professional interface using Ant Design 5.x
- **Real-time Data**: Live updates every 3 seconds
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Rich Visualizations**: Statistics, progress bars, tables, tags
- **Architecture Panel**: Complete architecture decisions visibility
- **Model Health Table**: Detailed model status with metrics
- **Task Progress**: Active task tracking with iteration counts

## 🚀 Quick Start

```bash
# Start the Ant Design dashboard
python scripts/run_dashboard.py

# Or with custom options
python scripts/run_dashboard.py --port 8888 --no-browser
```

Open http://127.0.0.1:8888 in your browser.

## 📸 Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│  ◈ Mission Control                                    Dashboard │
├─────────────────────────────────────────────────────────────────┤
│  [Metrics Cards]                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 2,871    │  │ $4.23    │  │ 158ms    │  │ 1        │        │
│  │ Calls    │  │ Cost     │  │ Latency  │  │ Projects │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────┐  ┌──────────────────────┐    │
│  │ 📋 Project Overview          │  │ 🏗️ Architecture      │    │
│  │                              │  │                      │    │
│  │ Build a scalable e-commerce  │  │ Style: Microservices │    │
│  │ API with microservices...    │  │ Paradigm: OOP        │    │
│  │                              │  │ API: REST            │    │
│  │ Progress: ██████████░░ 67%   │  │                      │    │
│  │ Tasks: 8/12 completed        │  │ [FastAPI] [Pydantic] │    │
│  │ Budget: $2.45 / $5.00        │  │ [PostgreSQL] [Redis] │    │
│  │                              │  │                      │    │
│  ├──────────────────────────────┤  │ Constraints:         │    │
│  │ ⚡ Active Task                │  │ ✓ Stateless services │    │
│  │                              │  │ ✓ JWT 24h expiry     │    │
│  │ task_009_payment_service     │  │                      │    │
│  │ Type: code_generation        │  │ Patterns:            │    │
│  │ Iteration: 2/3               │  │ ◈ CQRS               │    │
│  │ Score: 87% ████████░░░       │  │ ◈ Event Sourcing     │    │
│  │                              │  │ ◈ Circuit Breaker    │    │
│  └──────────────────────────────┘  └──────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  🤖 Model Status                                    [Refresh]   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Model        │ Provider │ Status      │ Success │ Latency │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ 👤 gpt-4o    │ openai   │ ✅ Available│ 98%     │ 125ms   │ │
│  │ 👤 deepseek  │ deepseek │ ✅ Available│ 96%     │ 180ms   │ │
│  │ 👤 kimi-k2.5 │ kimi     │ ❌ Unavail  │ -       │ -       │ │
│  │ 👤 gemini    │ google   │ ✅ Available│ 94%     │ 210ms   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🧩 Components

### 1. Metrics Cards
Four key metrics displayed at the top:
- **Total Calls**: Total API calls made
- **Total Cost**: Accumulated cost in USD
- **Avg Latency**: Average response latency
- **Active Projects**: Number of running projects

### 2. Project Card
Project overview with:
- Project description
- Real-time progress bar
- Task completion count
- Budget usage
- Elapsed time

### 3. Active Task Card
Current task details:
- Task ID and type
- Model being used
- Iteration counter
- Score visualization
- Prompt preview (expandable)

### 4. Architecture Card
Complete architecture information:
- Style, paradigm, API style
- Technology stack badges
- Constraints list
- Design patterns

### 5. Models Table
Detailed model status table:
- Avatar with availability indicator
- Provider information
- Availability status with reason tooltip
- Success rate percentage
- Average latency
- Total call count
- Cost per 1M tokens

## 🎨 UI Framework

### Ant Design Components Used

| Component | Usage |
|-----------|-------|
| Layout | Page structure (header, content, footer) |
| Card | Information panels |
| Statistic | Numeric displays |
| Progress | Progress bars and rings |
| Table | Model status table |
| Tag | Badges for technologies |
| Timeline | Future: Event timeline |
| Descriptions | Key-value displays |
| Alert | Notifications |
| Badge | Status indicators |
| Menu | Navigation |

### Icons

All icons from `@ant-design/icons`:
- DashboardOutlined
- ApiOutlined
- CodeOutlined
- ClusterOutlined
- CheckCircleOutlined
- CloseCircleOutlined
- ReloadOutlined
- And more...

## 🔄 Auto-Refresh

The dashboard automatically refreshes every 3 seconds:

```javascript
useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
}, []);
```

## 📡 API Endpoints

```
GET /api/status       # Complete dashboard state
GET /api/project      # Project information
GET /api/architecture # Architecture decisions
GET /api/active-task  # Currently active task
GET /api/models       # Model status list
GET /api/metrics      # System metrics
```

## 🎯 Color Scheme

| Color | Usage |
|-------|-------|
| `#1890ff` | Primary blue - links, buttons |
| `#52c41a` | Success green - available, passed |
| `#ff4d4f` | Error red - unavailable, failed |
| `#faad14` | Warning yellow - degraded |
| `#722ed1` | Purple - databases |
| `#13c2c2` | Cyan - frameworks |
| `#f0f2f5` | Background gray |
| `#001529` | Header dark blue |

## 📱 Responsive Design

The dashboard uses Ant Design's responsive grid:

```jsx
<Row gutter={24}>
    <Col xs={24} lg={16}>  {/* Full width on mobile, 2/3 on desktop */}
        <ProjectCard />
    </Col>
    <Col xs={24} lg={8}>   {/* Full width on mobile, 1/3 on desktop */}
        <ArchitectureCard />
    </Col>
</Row>
```

Breakpoints:
- **xs**: < 576px (mobile)
- **sm**: ≥ 576px (tablet)
- **md**: ≥ 768px
- **lg**: ≥ 992px (desktop)
- **xl**: ≥ 1200px
- **xxl**: ≥ 1600px

## 🔧 Customization

### Change Theme

Edit the style section in `dashboard_antd.py`:

```css
:root {
    --primary-color: #1890ff;
    --success-color: #52c41a;
    --error-color: #ff4d4f;
}
```

### Add New Card

```jsx
function CustomCard({ data }) {
    return (
        <Card title="Custom Card">
            <Statistic value={data.value} />
        </Card>
    );
}
```

### Change Refresh Interval

```javascript
const interval = setInterval(fetchData, 5000);  // 5 seconds
```

## 📦 Dependencies

Loaded via CDN (no installation needed):

```html
<!-- Ant Design CSS -->
<link rel="stylesheet" href="https://unpkg.com/antd@5.12.0/dist/antd.min.css">

<!-- React -->
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>

<!-- Ant Design JS -->
<script src="https://unpkg.com/antd@5.12.0/dist/antd.min.js"></script>

<!-- Ant Design Icons -->
<script src="https://unpkg.com/@ant-design/icons@5.2.6/dist/index.umd.min.js"></script>
```

## 🚀 Future Enhancements

- [ ] Charts with Recharts
- [ ] Dark mode toggle
- [ ] Multiple project tabs
- [ ] Real-time logs streaming
- [ ] Export to PDF/Excel
- [ ] User preferences
- [ ] WebSocket support

## 📝 Changelog

### v3.0 (Current)
- ✅ Ant Design UI
- ✅ Real-time updates
- ✅ Architecture panel
- ✅ Model status table
- ✅ Responsive design

### v2.0
- Custom CSS dashboard
- Enhanced data visibility

### v1.0
- Basic dashboard
- Static data
