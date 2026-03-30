"""
Adobe InDesign Plugin Development Rules
========================================
Comprehensive rules for InDesign plugin and script development.

Technology Choices:
- UXP (JavaScript/TypeScript) - Recommended for most modern use cases
- C++ (Adobe InDesign SDK) - Only for deep native access, custom hooks, high performance

Usage:
    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()
    config = rules.generate_config("My Plugin", technology="uxp")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class InDesignRulesConfig:
    """InDesign Plugin Rules Configuration."""
    plugin_name: str
    plugin_id: str
    version: str = "1.0.0"
    author: str = ""

    # Technology choice
    technology: str = "uxp"  # uxp, cpp, extendscript

    # UXP specific
    uxp_type: str = "panel"  # panel, modal, headless
    use_typescript: bool = True
    use_react: bool = True

    # C++ specific
    cpp_standard: str = "c++17"
    use_raii: bool = True
    use_sanitizers: bool = True

    # Distribution
    distribution_target: str = "marketplace"  # marketplace, enterprise, internal

    # Features
    include_telemetry: bool = True
    telemetry_opt_in: bool = True
    include_migration: bool = True
    include_ci: bool = True

    # Compliance
    gdpr_compliant: bool = True
    secure_credentials: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "plugin_id": self.plugin_id,
            "version": self.version,
            "technology": self.technology,
        }


class InDesignPluginRules:
    """
    Adobe InDesign Plugin Development Rules Engine.

    Provides comprehensive rules for professional InDesign plugin development
    using UXP (recommended) or C++ SDK.
    """

    # ═══════════════════════════════════════════════════════════════════
    # TECHNOLOGY PATHS
    # ═══════════════════════════════════════════════════════════════════

    TECHNOLOGY_PATHS = {
        "uxp": {
            "name": "UXP (Universal Extensibility Platform)",
            "description": "Modern JavaScript/TypeScript framework for Adobe Creative Cloud. Recommended for most use cases.",
            "recommended_for": [
                "Modern UI with HTML/CSS/JS",
                "Cross-platform compatibility",
                "Rapid development cycles",
                "Integration with Creative Cloud",
                "Most business logic implementations",
                "Network operations and APIs",
            ],
            "pros": [
                "Modern web technologies (React, TypeScript)",
                "Secure sandboxed environment",
                "Automatic Creative Cloud integration",
                "Cross-platform (Win/Mac)",
                "Easy distribution via Marketplace",
                "Hot reload during development",
                "Rich UI capabilities",
            ],
            "cons": [
                "Limited deep native access",
                "Some DOM operations restricted",
                "Performance overhead vs native",
                "Requires modern InDesign versions",
            ],
            "requirements": {
                "indesign": "2021 or later",
                "runtime": "UXP 6.0+",
                "languages": ["JavaScript", "TypeScript"],
            },
            "structure": [
                "manifest.json           # Plugin manifest",
                "package.json           # NPM dependencies",
                "tsconfig.json          # TypeScript config",
                "src/",
                "  ├── components/      # React/UI components",
                "  ├── controllers/     # Business logic",
                "  ├── utils/          # Utilities",
                "  └── index.ts        # Entry point",
                "ui/                    # HTML templates",
                "icons/                 # Plugin icons",
                "dist/                  # Build output",
            ],
        },
        "cpp": {
            "name": "C++ SDK (Native Plugin)",
            "description": "Low-level C++ SDK for deep native access, custom hooks, and maximum performance.",
            "recommended_for": [
                "Deep native InDesign access",
                "Custom event hooks",
                "High-performance operations",
                "Large document processing",
                "Custom file format support",
                "Low-level DOM manipulation",
            ],
            "pros": [
                "Full native access",
                "Maximum performance",
                "Custom event hooks",
                "Direct DOM manipulation",
                "Legacy InDesign support",
            ],
            "cons": [
                "Complex development",
                "Platform-specific builds (Win/Mac)",
                "Memory management complexity",
                "Harder to debug",
                "Longer development cycles",
                "Requires C++ expertise",
            ],
            "requirements": {
                "indesign": "2019 or later (depending on SDK)",
                "sdk": "Adobe InDesign SDK",
                "compiler": "Visual Studio (Win) / Xcode (Mac)",
            },
            "structure": [
                "Source/",
                "  ├── Plugin/          # Main plugin code",
                "  ├── Public/          # Public interfaces",
                "  └── Utils/          # Utilities",
                "Headers/               # Header files",
                "Resources/             # Resources",
                "Build/                 # Build scripts",
                "CMakeLists.txt        # CMake config",
            ],
        },
        "extendscript": {
            "name": "ExtendScript (Legacy)",
            "description": "Legacy JavaScript for older InDesign versions. Not recommended for new projects.",
            "recommended_for": [
                "Legacy InDesign support (pre-2021)",
                "Simple automation scripts",
                "Quick prototypes",
            ],
            "pros": [
                "Works with older InDesign",
                "Simple for basic scripts",
                "No build process",
            ],
            "cons": [
                "Deprecated technology",
                "No modern UI capabilities",
                "Limited security",
                "No Marketplace distribution",
                "Will be phased out",
            ],
            "note": "⚠️  ExtendScript is deprecated. Use UXP for new projects.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════
    # UXP BEST PRACTICES
    # ═══════════════════════════════════════════════════════════════════

    UXP_BEST_PRACTICES = """
## UXP (JavaScript/TypeScript) Best Practices

### Architecture Separation
```typescript
// ✅ SEPARATE concerns: UI, Logic, Native

// 1. UI Layer (React components)
// src/components/Panel.tsx
import React from 'react';
import { useController } from '../controllers/DocumentController';

export const Panel: React.FC = () => {
    const { documents, loadDocument } = useController();

    return (
        <div>
            {documents.map(doc => (
                <DocumentItem key={doc.id} doc={doc} />
            ))}
        </div>
    );
};

// 2. Business Logic Layer (Controllers)
// src/controllers/DocumentController.ts
export class DocumentController {
    async processDocument(docId: string): Promise<void> {
        // Business logic here
        const data = await this.fetchData(docId);
        await this.nativeLayer.applyChanges(data);
    }
}

// 3. Native Layer (InDesign API)
// src/native/DocumentApi.ts
export class DocumentApi {
    async applyChanges(data: DocumentData): Promise<void> {
        // Direct InDesign API calls
        await require('indesign').app.activeDocument.applyChanges(data);
    }
}
```

### TypeScript Strict Mode
```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "CommonJS",
    "lib": ["ES2020", "DOM"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

### Non-Blocking Operations
```typescript
// ❌ NEVER block the main thread
function processLargeDocument() {
    for (let i = 0; i < 10000; i++) {
        // Blocking!
        processPage(i);
    }
}

// ✅ ALWAYS use async/await or batch processing
async function processLargeDocumentBatched(): Promise<void> {
    const batchSize = 100;
    const totalPages = 10000;

    for (let i = 0; i < totalPages; i += batchSize) {
        await processBatch(i, Math.min(i + batchSize, totalPages));

        // Yield to UI
        await new Promise(resolve => setTimeout(resolve, 0));

        // Update progress
        updateProgress((i / totalPages) * 100);
    }
}

// ✅ Use Web Workers for heavy computation
// (UXP supports limited worker-like functionality)
```

### Secure Credential Management
```typescript
// ❌ NEVER hardcode credentials
const API_KEY = "sk-1234567890abcdef"; // NEVER!

// ✅ Use secure storage
import { secureStorage } from 'uxp';

async function storeCredentials(apiKey: string): Promise<void> {
    await secureStorage.setItem('api_key', apiKey);
}

async function getCredentials(): Promise<string | null> {
    return await secureStorage.getItem('api_key');
}

// ✅ Use OAuth when available
async function authenticateWithOAuth(): Promise<void> {
    const oauth = require('oauth');
    const token = await oauth.authenticate({
        clientId: 'your-client-id',
        scope: ['read', 'write'],
    });
    await secureStorage.setItem('oauth_token', token);
}
```

### Error Handling
```typescript
// ✅ Always handle errors gracefully
try {
    await processDocument();
} catch (error) {
    if (error instanceof InDesignError) {
        // Handle InDesign-specific errors
        showErrorDialog(`InDesign Error: ${{error.message}}`);
    } else if (error instanceof NetworkError) {
        // Handle network errors
        showErrorDialog('Network connection failed. Please check your connection.');
    } else {
        // Unknown error
        console.error('Unexpected error:', error);
        showErrorDialog('An unexpected error occurred.');

        // Report telemetry (if user opted in)
        if (telemetry.isEnabled()) {
            telemetry.reportError(error);
        }
    }
}
```

### Memory Management
```typescript
// ✅ Clean up references
class DocumentProcessor {
    private activeDocuments: Map<string, Document> = new Map();

    async process(docId: string): Promise<void> {
        const doc = await this.loadDocument(docId);
        this.activeDocuments.set(docId, doc);

        try {
            await this.performOperations(doc);
        } finally {
            // Always clean up
            this.activeDocuments.delete(docId);
            doc.close();
        }
    }
}

// ✅ Use weak references where appropriate
const cache = new WeakMap<Document, ProcessedData>();
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # C++ SDK BEST PRACTICES
    # ═══════════════════════════════════════════════════════════════════

    CPP_BEST_PRACTICES = """
## C++ SDK Best Practices

### RAII (Resource Acquisition Is Initialization)
```cpp
// ✅ Use RAII for all resources
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

    // Disable copy
    DocumentRAII(const DocumentRAII&) = delete;
    DocumentRAII& operator=(const DocumentRAII&) = delete;

    // Enable move
    DocumentRAII(DocumentRAII&& other) noexcept : doc(other.doc) {
        other.doc = nullptr;
    }
};

// Usage - automatically cleaned up
void processDocument() {
    DocumentRAII doc(GetActiveDocument());
    // Process...
    // Automatically released when doc goes out of scope
}
```

### Memory Sanitizers
```cmake
# CMakeLists.txt
# Debug builds with sanitizers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Address Sanitizer
    target_compile_options(plugin PRIVATE -fsanitize=address)
    target_link_options(plugin PRIVATE -fsanitize=address)

    # Memory Sanitizer (Linux only)
    if(LINUX)
        target_compile_options(plugin PRIVATE -fsanitize=memory)
        target_link_options(plugin PRIVATE -fsanitize=memory)
    endif()

    # Undefined Behavior Sanitizer
    target_compile_options(plugin PRIVATE -fsanitize=undefined)
    target_link_options(plugin PRIVATE -fsanitize=undefined)
endif()
```

### Non-Blocking Operations
```cpp
// ✅ Use threading for heavy operations
#include <thread>
#include <future>

class AsyncDocumentProcessor {
public:
    std::future<void> processAsync(IDocument* doc) {
        return std::async(std::launch::async, [doc]() {
            // Heavy processing in background thread
            processDocumentHeavy(doc);
        });
    }
};

// ✅ Use InDesign's idle tasks for UI updates
void ScheduleIdleUpdate() {
    InterfacePtr<ISession> session(GetExecutionContextSession());
    session->ScheduleIdleTask(
        kPluginID,
        [this]() { this->UpdateProgressUI(); }
    );
}
```

### Error Handling
```cpp
// ✅ Always check return codes
ErrorCode result = document->ProcessData();
if (result != kSuccess) {
    // Log error
    LOG_ERROR("Document processing failed: {}", ErrorUtils::GetErrorString(result));

    // Show user-friendly message
    ShowAlert("Processing failed. Please try again or contact support.");

    // Cleanup
    RollbackChanges();
    return kFailure;
}
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # SECURITY RULES
    # ═══════════════════════════════════════════════════════════════════

    SECURITY_RULES = """
## Security Rules (CRITICAL)

### Credential Management
```typescript
// UXP - Secure Storage
import { secureStorage } from 'uxp';

// Store encrypted
await secureStorage.setItem('api_key', apiKey);

// Retrieve
const apiKey = await secureStorage.getItem('api_key');

// Never log credentials
console.log('API Key:', apiKey); // ❌ NEVER!
```

### GDPR Compliance
```typescript
// ✅ Explicit consent
interface PrivacySettings {
    telemetryEnabled: boolean;
    dataCollection: boolean;
    marketingEmails: boolean;
}

// Show privacy dialog on first run
async function showPrivacyDialog(): Promise<PrivacySettings> {
    const settings = await openDialog({
        title: 'Privacy Settings',
        content: `
            <h2>Data Collection Consent</h2>
            <p>We would like to collect:</p>
            <ul>
                <li>Error reports (to improve stability)</li>
                <li>Usage statistics (anonymized)</li>
            </ul>
            <p>You can change these settings at any time.</p>
        `,
        buttons: ['Accept', 'Decline'],
    });

    return {
        telemetryEnabled: settings.result === 'Accept',
        dataCollection: settings.result === 'Accept',
        marketingEmails: false,
    };
}

// Check before any data collection
async function reportTelemetry(data: TelemetryData): Promise<void> {
    const settings = await loadPrivacySettings();
    if (!settings.telemetryEnabled) {
        return; // Respect user choice
    }

    // Anonymize data
    const anonymized = anonymizeData(data);
    await sendToTelemetryServer(anonymized);
}

// Right to be forgotten
async function deleteUserData(): Promise<void> {
    await clearLocalStorage();
    await clearSecureStorage();
    await clearCache();
    await notifyServerOfDeletion();
}
```

### Permission Model
```json
// manifest.json - Explicit permissions
{
  "requiredPermissions": {
    "allowCodeGenerationFromStrings": false,
    "launchProcess": {
      "schemes": ["https"],
      "domains": ["api.example.com"]
    },
    "network": {
      "domains": ["api.example.com", "cdn.example.com"]
    },
    "clipboard": {
      "read": true,
      "write": true
    },
    "fileSystem": {
      "read": [".indd", ".idml", ".pdf"],
      "write": [".pdf", ".png"]
    }
  }
}
```

### Input Validation
```typescript
// ✅ Always validate file paths
function validateFilePath(filePath: string): boolean {
    // Prevent directory traversal
    if (filePath.includes('..')) {
        return false;
    }

    // Whitelist allowed extensions
    const allowedExtensions = ['.indd', '.idml', '.pdf', '.png'];
    const ext = path.extname(filePath).toLowerCase();
    return allowedExtensions.includes(ext);
}

// ✅ Validate document content
async function validateDocument(doc: Document): Promise<boolean> {
    // Check for suspicious content
    if (doc.pages.length > 10000) {
        throw new Error('Document too large');
    }

    // Scan for malicious links
    const links = await doc.getLinks();
    for (const link of links) {
        if (isMaliciousUrl(link.url)) {
            throw new Error('Malicious link detected');
        }
    }

    return true;
}
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # TESTING & STRESS CASES
    # ═══════════════════════════════════════════════════════════════════

    TESTING_RULES = """
## Testing & Stress Cases

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: InDesign Plugin CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, macos-latest]
        indesign: ['2022', '2023', '2024']

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install dependencies
      run: npm ci

    - name: Lint
      run: npm run lint

    - name: Type Check
      run: npm run type-check

    - name: Build
      run: npm run build

    - name: Smoke Test
      run: npm run test:smoke
      env:
        INDESIGN_VERSION: ${{ matrix.indesign }}

    - name: Package
      run: npm run package
```

### Memory Testing (C++)
```cpp
// Memory stress test
TEST_F(DocumentTest, MemoryStressTest) {
    // Open 100 large documents
    for (int i = 0; i < 100; i++) {
        auto doc = OpenDocument("large_file_" + std::to_string(i) + ".indd");

        // Process
        ProcessDocument(doc);

        // Check memory hasn't leaked
        auto memoryUsage = GetMemoryUsage();
        EXPECT_LT(memoryUsage, 1024 * 1024 * 1024); // < 1GB

        // Close and verify cleanup
        CloseDocument(doc);

        // Force garbage collection
        CollectGarbage();
    }
}
```

### Document Stress Testing
```typescript
// Test with various document types
const stressTestDocuments = [
    { name: 'empty.indd', pages: 0, expected: 'should handle gracefully' },
    { name: 'small.indd', pages: 10, expected: 'should process quickly' },
    { name: 'medium.indd', pages: 100, expected: 'should process in < 5s' },
    { name: 'large.indd', pages: 1000, expected: 'should process in < 30s' },
    { name: 'huge.indd', pages: 10000, expected: 'should not crash' },
    { name: 'corrupted.indd', pages: -1, expected: 'should show error dialog' },
    { name: 'missing_fonts.indd', pages: 50, expected: 'should handle missing fonts' },
    { name: 'complex_layouts.indd', pages: 200, expected: 'should preserve layouts' },
    { name: 'many_links.indd', pages: 50, links: 10000, expected: 'should handle many links' },
];

describe('Stress Tests', () => {
    for (const doc of stressTestDocuments) {
        test(`should handle ${doc.name}`, async () => {
            const startTime = Date.now();

            await expect(processDocument(doc.name))
                .resolves.not.toThrow();

            const duration = Date.now() - startTime;
            console.log(`${doc.name} processed in ${duration}ms`);
        }, 60000); // 60s timeout for large docs
    }
});
```

### Profiling
```typescript
// Performance profiling
class PerformanceProfiler {
    private markers: Map<string, number> = new Map();

    start(marker: string): void {
        this.markers.set(marker, performance.now());
    }

    end(marker: string): number {
        const start = this.markers.get(marker);
        if (!start) return 0;

        const duration = performance.now() - start;
        console.log(`${marker}: ${duration.toFixed(2)}ms`);
        return duration;
    }
}

// Usage
const profiler = new PerformanceProfiler();

profiler.start('documentLoad');
const doc = await loadDocument('large.indd');
profiler.end('documentLoad');

profiler.start('processing');
await processDocument(doc);
profiler.end('processing');
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # DISTRIBUTION & MIGRATION
    # ═══════════════════════════════════════════════════════════════════

    DISTRIBUTION_RULES = """
## Distribution & Migration

### Versioned Migrations
```typescript
// Migration system
interface Migration {
    version: string;
    migrate: (oldData: any) => Promise<any>;
}

const migrations: Migration[] = [
    {
        version: '1.1.0',
        migrate: async (data: any) => {
            // Migrate from 1.0.x to 1.1.0
            return {
                ...data,
                newField: data.oldField || 'default',
            };
        },
    },
    {
        version: '2.0.0',
        migrate: async (data: any) => {
            // Breaking changes - major migration
            return transformDataStructure(data);
        },
    },
];

async function migrateSettings(currentVersion: string, targetVersion: string): Promise<void> {
    const settings = await loadSettings();
    let migratedData = settings;

    for (const migration of migrations) {
        if (compareVersions(currentVersion, migration.version) < 0 &&
            compareVersions(targetVersion, migration.version) >= 0) {
            try {
                migratedData = await migration.migrate(migratedData);
            } catch (error) {
                // Rollback on failure
                await rollbackMigration();
                throw new Error(`Migration to ${migration.version} failed`);
            }
        }
    }

    await saveSettings(migratedData);
}
```

### Packaging for Marketplace
```json
// manifest.json - Marketplace ready
{
  "id": "com.example.myplugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "host": [
    {
      "app": "ID",
      "minVersion": "16.0",
      "maxVersion": "19.0"
    }
  ],
  "type": "panel",
  "uiEntry": {
    "type": "panel",
    "label": "My Plugin"
  },
  "icons": [
    { "width": 23, "height": 23, "path": "icons/icon-23.png" },
    { "width": 48, "height": 48, "path": "icons/icon-48.png" }
  ],
  "requiredPermissions": {
    "network": {
      "domains": ["api.example.com"]
    }
  }
}
```

### Signing (C++)
```bash
# Windows - Sign with certificate
signtool.exe sign /f certificate.pfx /p password /t http://timestamp.digicert.com plugin.dll

# macOS - Sign with Developer ID
codesign --force --sign "Developer ID Application: Your Name" --timestamp plugin.bundle

# Notarize (macOS)
xcrun altool --notarize-app --primary-bundle-id "com.example.plugin" \
    --username "your@email.com" --password "@keychain:AC_PASSWORD" \
    --file plugin.zip
```
"""

    def __init__(self):
        """Initialize rules engine."""
        pass

    def get_technology_path(self, tech_key: str) -> dict[str, Any]:
        """Get technology path details."""
        return self.TECHNOLOGY_PATHS.get(tech_key, {})

    def get_all_technologies(self) -> dict[str, dict[str, Any]]:
        """Get all available technologies."""
        return self.TECHNOLOGY_PATHS

    def recommend_technology(self,
                            requires_native_access: bool = False,
                            requires_high_performance: bool = False,
                            requires_custom_hooks: bool = False,
                            target_indesign_version: str = "2024",
                            team_expertise: str = "javascript") -> str:
        """
        Recommend technology based on requirements.

        Args:
            requires_native_access: Needs deep InDesign access
            requires_high_performance: Performance-critical
            requires_custom_hooks: Custom event hooks needed
            target_indesign_version: Minimum InDesign version
            team_expertise: Team's primary language
        """
        # Check for C++ requirements
        if requires_native_access or requires_high_performance or requires_custom_hooks:
            return "cpp"

        # Check InDesign version
        version_year = int(target_indesign_version)
        if version_year < 2021:
            return "extendscript"  # Legacy support

        # Default to UXP for modern development
        return "uxp"

    def generate_config(self,
                       plugin_name: str,
                       technology: str | None = None,
                       **kwargs) -> InDesignRulesConfig:
        """
        Generate InDesign plugin configuration.

        Args:
            plugin_name: Human-readable plugin name
            technology: uxp, cpp, or extendscript
            **kwargs: Additional configuration options
        """
        # Generate plugin ID
        plugin_id = plugin_name.lower().replace(' ', '-').replace('_', '-')

        # Determine technology
        if technology is None:
            technology = self.recommend_technology(
                requires_native_access=kwargs.get('requires_native_access', False),
                requires_high_performance=kwargs.get('requires_high_performance', False),
                requires_custom_hooks=kwargs.get('requires_custom_hooks', False),
                target_indesign_version=kwargs.get('target_indesign_version', '2024'),
            )

        return InDesignRulesConfig(
            plugin_name=plugin_name,
            plugin_id=plugin_id,
            version=kwargs.get('version', '1.0.0'),
            author=kwargs.get('author', ''),
            technology=technology,
            uxp_type=kwargs.get('uxp_type', 'panel'),
            use_typescript=kwargs.get('use_typescript', True),
            use_react=kwargs.get('use_react', True),
            cpp_standard=kwargs.get('cpp_standard', 'c++17'),
            use_raii=kwargs.get('use_raii', True),
            use_sanitizers=kwargs.get('use_sanitizers', True),
            distribution_target=kwargs.get('distribution_target', 'marketplace'),
            include_telemetry=kwargs.get('include_telemetry', True),
            telemetry_opt_in=kwargs.get('telemetry_opt_in', True),
            include_migration=kwargs.get('include_migration', True),
            include_ci=kwargs.get('include_ci', True),
            gdpr_compliant=kwargs.get('gdpr_compliant', True),
            secure_credentials=kwargs.get('secure_credentials', True),
        )

    def get_rules_file_content(self, config: InDesignRulesConfig) -> str:
        """Generate .ai-rules.md content for InDesign plugin."""
        tech_info = self.get_technology_path(config.technology)

        # Select appropriate best practices
        if config.technology == "uxp":
            best_practices = self.UXP_BEST_PRACTICES
        elif config.technology == "cpp":
            best_practices = self.CPP_BEST_PRACTICES
        else:
            best_practices = "## ExtendScript (Legacy)\n\n⚠️  ExtendScript is deprecated. Migrate to UXP.\n"

        content = f"""# InDesign Plugin Rules: {config.plugin_name}

## 🎯 Technology Decision
**Selected**: {tech_info.get('name', config.technology)}

**Description**: {tech_info.get('description', '')}

**Pros**:
{chr(10).join(['- ' + p for p in tech_info.get('pros', [])])}

**Cons**:
{chr(10).join(['- ' + c for c in tech_info.get('cons', [])])}

## 📋 Plugin Configuration
- **Plugin Name**: {config.plugin_name}
- **Plugin ID**: `{config.plugin_id}`
- **Version**: {config.version}
- **Technology**: {config.technology.upper()}
- **Distribution**: {config.distribution_target}

## 🚨 MUST FOLLOW RULES

### 1. Architecture Separation
✅ SEPARATE: UI, Business Logic, Native Modules
- UI Layer: React components (UXP) or dialogs (C++)
- Logic Layer: Controllers/Services
- Native Layer: InDesign API calls

### 2. Non-Blocking Operations
✅ NEVER block the main thread
- Use async/await for operations
- Batch process large documents
- Show progress indicators
- Yield to UI periodically

### 3. Security (CRITICAL)
✅ Secure credential management
- Use secureStorage (UXP) or keychain (C++)
- OAuth for authentication
- NEVER hardcode API keys
- GDPR compliance required

### 4. Memory Management
✅ Clean up resources
- Use RAII (C++)
- Clear references (UXP)
- Close documents when done
- Test for memory leaks

### 5. Testing Requirements
✅ Comprehensive testing
- Smoke tests on multiple InDesign versions
- Stress tests with large/corrupted files
- Memory leak detection (C++)
- CI/CD pipeline required

{best_practices}

{self.SECURITY_RULES}

{self.TESTING_RULES}

{self.DISTRIBUTION_RULES}

## ✅ Development Checklist

### Phase 1: Setup (15 min)
- [ ] Choose technology (UXP recommended)
- [ ] Setup project structure
- [ ] Configure TypeScript strict mode (UXP)
- [ ] Setup CMake/Visual Studio (C++)
- [ ] Configure CI/CD pipeline
- [ ] Setup linting (ESLint/TS or C++ lint)

### Phase 2: Architecture (30 min)
- [ ] Separate UI, logic, native layers
- [ ] Implement async/non-blocking operations
- [ ] Setup secure credential storage
- [ ] Configure GDPR consent
- [ ] Setup telemetry (opt-in)

### Phase 3: Development (varies)
- [ ] Implement core functionality
- [ ] Handle errors gracefully
- [ ] Add progress indicators
- [ ] Optimize for large documents
- [ ] Memory profiling (C++)

### Phase 4: Security (20 min) ⚠️ CRITICAL
- [ ] Secure credential management
- [ ] Input validation
- [ ] GDPR compliance
- [ ] Permission model in manifest
- [ ] Data encryption

### Phase 5: Testing (30 min)
- [ ] Unit tests
- [ ] Smoke tests (multiple ID versions)
- [ ] Stress tests (large/corrupted files)
- [ ] Memory leak detection
- [ ] Performance profiling

### Phase 6: Distribution (20 min)
- [ ] Version migration system
- [ ] Code signing (C++)
- [ ] Marketplace packaging
- [ ] Documentation
- [ ] Changelog

## 📖 References
- [Adobe UXP Documentation](https://developer.adobe.com/uxp/)
- [InDesign Scripting Guide](https://www.adobe.com/devnet/indesign/documentation.html)
- [InDesign SDK](https://www.adobe.com/devnet/indesign/sdk.html)
- [Creative Cloud Marketplace](https://exchange.adobe.com/)
"""
        return content

    def save_rules_file(self, config: InDesignRulesConfig, output_dir: Path) -> Path:
        """Save rules file to output directory."""
        rules_content = self.get_rules_file_content(config)
        rules_file = output_dir / ".ai-rules.md"
        rules_file.write_text(rules_content, encoding="utf-8")
        return rules_file


# Convenience function
def generate_indesign_plugin_rules(
    plugin_name: str,
    output_dir: Path,
    **kwargs
) -> Path:
    """
    Generate InDesign plugin rules file.

    Args:
        plugin_name: Name of the plugin
        output_dir: Directory to save rules file
        **kwargs: Additional configuration options

    Returns:
        Path to generated rules file
    """
    rules = InDesignPluginRules()
    config = rules.generate_config(plugin_name, **kwargs)
    return rules.save_rules_file(config, output_dir)


if __name__ == "__main__":
    # Demo
    rules = InDesignPluginRules()

    print("=" * 70)
    print("Adobe InDesign Plugin Development Rules")
    print("=" * 70)

    print("\n📚 Technology Paths:")
    for key, path in rules.get_all_technologies().items():
        print(f"\n  {path['name']} ({key})")
        print(f"    {path['description'][:80]}...")

    print("\n\n🎯 Recommendation Examples:")

    # UXP recommendation
    rec = rules.recommend_technology(
        requires_native_access=False,
        target_indesign_version="2024",
    )
    print(f"  Standard plugin → {rec.upper()}")

    # C++ recommendation
    rec = rules.recommend_technology(
        requires_native_access=True,
        requires_high_performance=True,
    )
    print(f"  High-performance → {rec.upper()}")

    print("\n\n📄 Generate Rules for 'Document Processor':")
    config = rules.generate_config("Document Processor", technology="uxp")
    print(f"  Plugin: {config.plugin_name}")
    print(f"  ID: {config.plugin_id}")
    print(f"  Technology: {config.technology.upper()}")
    print(f"  TypeScript: {config.use_typescript}")
    print(f"  React: {config.use_react}")

    print("\n\n✅ InDesign Plugin Rules Engine Ready!")
