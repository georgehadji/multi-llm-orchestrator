# Adobe InDesign Plugin Development Rules

Professional rules engine for InDesign plugin development using UXP or C++ SDK.

## 🎯 Technology Choices

### UXP (Universal Extensibility Platform) - RECOMMENDED
**For**: Modern UI, rapid development, Creative Cloud integration

```
my-plugin/
├── manifest.json       # Plugin manifest
├── package.json        # NPM dependencies
├── tsconfig.json       # TypeScript strict
├── src/
│   ├── components/    # React components
│   ├── controllers/   # Business logic
│   └── utils/        # Utilities
├── ui/               # HTML templates
└── dist/             # Build output
```

**Requirements:**
- InDesign 2021 or later
- UXP 6.0+
- Node.js 18+

### C++ SDK (Native)
**For**: Deep native access, high performance, custom hooks

```
my-plugin/
├── Source/
│   ├── Plugin/       # Main plugin code
│   ├── Public/       # Public interfaces
│   └── Utils/       # Utilities
├── Headers/          # Headers
└── CMakeLists.txt   # CMake config
```

**Requirements:**
- InDesign 2019 or later
- Visual Studio (Win) / Xcode (Mac)
- Adobe InDesign SDK

### ExtendScript (Legacy) ⚠️
**Status**: Deprecated, not recommended for new projects

## 🚀 Quick Start

### Generate Rules for UXP Plugin

```python
from orchestrator.indesign_plugin_rules import (
    InDesignPluginRules,
    generate_indesign_plugin_rules,
)

# Method 1: Using the function
rules_file = generate_indesign_plugin_rules(
    plugin_name="Document Automation",
    output_dir=Path("./output"),
    technology="uxp",
    use_typescript=True,
    use_react=True,
    distribution_target="marketplace",
)

# Method 2: Using the class
rules = InDesignPluginRules()
config = rules.generate_config(
    plugin_name="Document Automation",
    technology="uxp",  # or auto-detect
    use_typescript=True,
    use_react=True,
    include_ci=True,
)
rules.save_rules_file(config, Path("./output"))
```

### Auto-Detect Technology

```python
# Automatically recommends best technology
config = rules.generate_config(
    plugin_name="High Performance Processor",
    requires_native_access=True,
    requires_high_performance=True,
    requires_custom_hooks=True,
)
# → Recommends: C++
```

## 🏗️ Architecture Patterns

### UXP: Separation of Concerns

```typescript
// 1. UI Layer (React)
export const Panel: React.FC = () => {
    const controller = useDocumentController();
    return <DocumentList docs={controller.documents} />;
};

// 2. Business Logic (Controller)
export class DocumentController {
    async processDocument(id: string): Promise<void> {
        const data = await this.fetchData(id);
        await this.nativeApi.applyChanges(data);
    }
}

// 3. Native Layer (InDesign API)
export class InDesignApi {
    async applyChanges(data: Data): Promise<void> {
        const indesign = require('indesign');
        await indesign.app.activeDocument.applyChanges(data);
    }
}
```

### C++: RAII Pattern

```cpp
// Resource management with RAII
class DocumentRAII {
private:
    IDocument* doc;
    
public:
    explicit DocumentRAII(IDocument* d) : doc(d) {
        if (doc) doc->AddRef();
    }
    
    ~DocumentRAII() {
        if (doc) doc->Release();
    }
    
    // Disable copy, enable move
    DocumentRAII(const DocumentRAII&) = delete;
    DocumentRAII(DocumentRAII&& other) noexcept : doc(other.doc) {
        other.doc = nullptr;
    }
};
```

## 🛡️ Security Requirements

### Credential Management
```typescript
// ✅ Use secure storage
import { secureStorage } from 'uxp';

// Store encrypted
await secureStorage.setItem('api_key', apiKey);

// Retrieve
const apiKey = await secureStorage.getItem('api_key');

// ❌ NEVER hardcode
const API_KEY = "sk-12345"; // NEVER!
```

### GDPR Compliance
```typescript
// ✅ Explicit consent
interface PrivacySettings {
    telemetryEnabled: boolean;
    dataCollection: boolean;
}

// Show on first run
const settings = await showPrivacyDialog();
// Store user choice
await savePrivacySettings(settings);

// Check before collecting
if (settings.telemetryEnabled) {
    await reportTelemetry(data);
}
```

## ⚡ Performance Guidelines

### Non-Blocking Operations
```typescript
// ❌ NEVER block main thread
for (let i = 0; i < 10000; i++) {
    processPage(i); // Blocking!
}

// ✅ Use async/await
async function processBatched(): Promise<void> {
    for (let i = 0; i < total; i += batchSize) {
        await processBatch(i, i + batchSize);
        await new Promise(r => setTimeout(r, 0)); // Yield
        updateProgress((i / total) * 100);
    }
}
```

### Memory Management (C++)
```cpp
// ✅ RAII for resources
{
    DocumentRAII doc(GetActiveDocument());
    ProcessDocument(doc);
    // Auto-released when doc goes out of scope
}

// ✅ Sanitizers in debug
// CMakeLists.txt
target_compile_options(plugin PRIVATE -fsanitize=address)
```

## 🧪 Testing Requirements

### Stress Testing
```typescript
const stressTests = [
    { name: 'empty.indd', pages: 0 },
    { name: 'large.indd', pages: 10000 },
    { name: 'corrupted.indd', pages: -1 },
    { name: 'many_links.indd', pages: 50, links: 10000 },
];

for (const test of stressTests) {
    test(`handles ${test.name}`, async () => {
        await expect(processDocument(test.name))
            .resolves.not.toThrow();
    }, 60000);
}
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
strategy:
  matrix:
    os: [windows-latest, macos-latest]
    indesign: ['2022', '2023', '2024']

steps:
  - name: Smoke Test
    run: npm run test:smoke
    env:
      INDESIGN_VERSION: ${{ matrix.indesign }}
```

## 📦 Distribution

### Versioned Migrations
```typescript
const migrations = [
    {
        version: '1.1.0',
        migrate: async (data) => ({
            ...data,
            newField: data.oldField,
        }),
    },
    {
        version: '2.0.0',
        migrate: async (data) => {
            // Breaking changes
            return transformData(data);
        },
    },
];

async function migrate(from: string, to: string): Promise<void> {
    for (const migration of migrations) {
        if (shouldApply(migration.version, from, to)) {
            try {
                data = await migration.migrate(data);
            } catch (error) {
                await rollback();
                throw error;
            }
        }
    }
}
```

### Code Signing (C++)
```bash
# Windows
signtool.exe sign /f cert.pfx /p pass /t http://timestamp.digicert.com plugin.dll

# macOS
codesign --force --sign "Developer ID" --timestamp plugin.bundle
```

## 🎨 Technology Decision Matrix

| Requirement | UXP | C++ | ExtendScript |
|-------------|-----|-----|--------------|
| Modern UI | ✅ | ⚠️ | ❌ |
| Cross-platform | ✅ | ❌ | ✅ |
| Deep native access | ⚠️ | ✅ | ❌ |
| High performance | ⚠️ | ✅ | ❌ |
| Rapid development | ✅ | ❌ | ✅ |
| Creative Cloud | ✅ | ⚠️ | ❌ |
| Legacy support | ❌ | ✅ | ✅ |
| Future-proof | ✅ | ✅ | ❌ |

## 📋 Development Checklist

### Phase 1: Setup
- [ ] Choose technology (UXP recommended)
- [ ] Setup project structure
- [ ] Configure TypeScript strict (UXP)
- [ ] Setup CMake (C++)
- [ ] Configure CI/CD

### Phase 2: Security ⚠️ CRITICAL
- [ ] Secure credential storage
- [ ] GDPR consent
- [ ] Input validation
- [ ] Permission model
- [ ] Data encryption

### Phase 3: Testing
- [ ] Unit tests
- [ ] Smoke tests (multiple ID versions)
- [ ] Stress tests (large files)
- [ ] Memory profiling (C++)
- [ ] Performance testing

### Phase 4: Distribution
- [ ] Migration system
- [ ] Code signing (C++)
- [ ] Marketplace packaging
- [ ] Documentation

## ✅ Generated Rules Include

- Architecture separation guidelines
- Security rules (credentials, GDPR)
- Performance best practices
- Testing requirements
- CI/CD configuration
- Distribution guidelines
- Migration strategies
- Full development checklist

## 📖 References

- [Adobe UXP](https://developer.adobe.com/uxp/)
- [InDesign Scripting](https://www.adobe.com/devnet/indesign/)
- [InDesign SDK](https://www.adobe.com/devnet/indesign/sdk.html)
- [Creative Cloud Marketplace](https://exchange.adobe.com/)
