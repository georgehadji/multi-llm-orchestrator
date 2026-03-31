"""
WordPress Plugin Development Rules
====================================
Comprehensive rules and best practices for WordPress plugin development.

Based on:
- WordPress Plugin Handbook
- WordPress PHP Coding Standards
- WordPress Security Best Practices
- WordPress.org Plugin Repository Guidelines

Usage:
    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    config = rules.generate_config(plugin_name="My Plugin")
    rules.save_rules_file(config, Path("./output"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class WPRulesConfig:
    """WordPress Plugin Rules Configuration."""

    plugin_name: str
    plugin_slug: str
    text_domain: str
    namespace: str
    prefix: str
    version: str = "1.0.0"
    author: str = ""
    license: str = "GPL-2.0+"

    # Architecture path
    architecture_path: str = "modular_oop"  # modular_oop, lightweight, headless

    # Features
    include_composer: bool = True
    include_tests: bool = True
    include_ci: bool = True
    include_i18n: bool = True
    include_rest: bool = False
    include_blocks: bool = False

    # Security level
    strict_security: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "plugin_slug": self.plugin_slug,
            "text_domain": self.text_domain,
            "namespace": self.namespace,
            "prefix": self.prefix,
            "version": self.version,
            "architecture_path": self.architecture_path,
        }


class WordPressPluginRules:
    """
    WordPress Plugin Development Rules Engine.

    Provides comprehensive rules, constraints, and best practices
    for professional WordPress plugin development.
    """

    # ═══════════════════════════════════════════════════════════════════
    # ARCHITECTURE PATHS
    # ═══════════════════════════════════════════════════════════════════

    ARCHITECTURE_PATHS = {
        "modular_oop": {
            "name": "Modular OOP + Composer + Tests",
            "description": "Production-ready plugin with namespaced classes, Composer autoloading, full test suite, and CI/CD.",
            "recommended_for": [
                "Public distribution on WordPress.org",
                "Team collaboration",
                "Long-term maintenance",
                "Complex functionality",
            ],
            "pros": [
                "High maintainability",
                "Scalability",
                "Easy code review",
                "Better quality",
                "PSR-4 autoloading",
                "Dependency injection support",
            ],
            "cons": [
                "Steeper learning curve for beginners",
                "Additional build/CI overhead",
                "Requires Composer knowledge",
            ],
            "structure": [
                "composer.json (PSR-4 autoload)",
                "src/ (namespaced classes)",
                "tests/ (PHPUnit + wp-env)",
                ".github/workflows/ (CI/CD)",
                "phpunit.xml",
                "phpcs.xml (WordPress standards)",
            ],
        },
        "lightweight": {
            "name": "Lightweight Procedural",
            "description": "Simple procedural code with strict prefixing, minimal dependencies, no Composer.",
            "recommended_for": [
                "Small internal projects",
                "Proof-of-concept",
                "Rapid MVP development",
                "Client projects with tight deadlines",
            ],
            "pros": [
                "Fast delivery",
                "Small learning curve",
                "No build process",
                "Easy to understand",
            ],
            "cons": [
                "Scalability issues",
                "Higher conflict risk",
                "Difficult to add tests later",
                "Code organization challenges",
            ],
            "structure": [
                "main-plugin.php (procedural)",
                "includes/ (functions)",
                "admin/ (admin screens)",
                "public/ (frontend)",
                "No Composer",
            ],
        },
        "headless": {
            "name": "Headless / Service-Oriented",
            "description": "Thin WordPress plugin that acts as REST client to external microservices.",
            "recommended_for": [
                "Heavy compute requirements",
                "Multi-platform integration",
                "SaaS + WP frontend",
                "Strict separation of concerns",
            ],
            "pros": [
                "Clean separation",
                "Scalable backend",
                "Easy reuse across platforms",
                "WP stays lightweight",
            ],
            "cons": [
                "Offline mode difficulties",
                "Additional ops/security surface",
                "Network latency issues",
                "External service dependency",
            ],
            "structure": [
                "src/ApiClient/ (external service client)",
                "src/Cache/ (response caching)",
                "src/Auth/ (authentication)",
                "Minimal local logic",
            ],
        },
    }

    # ═══════════════════════════════════════════════════════════════════
    # CODING STANDARDS
    # ═══════════════════════════════════════════════════════════════════

    CODING_STANDARDS = """
## WordPress PHP Coding Standards

### Indentation
- Use TABS, not spaces
- Tab size: 4 (configurable in editor)

### Naming Conventions
- Classes: `Class_Name` (uppercase words, underscores)
- Methods: `method_name()` (lowercase, underscores)
- Functions: `prefix_function_name()` (always prefix!)
- Variables: `$variable_name` (lowercase, underscores)
- Constants: `CONSTANT_NAME` (uppercase, underscores)
- Files: `class-class-name.php`, `template-name.php`
- Hooks: `prefix_action_name`, `prefix_filter_name`

### Namespacing (for OOP)
```php
namespace Vendor\\PluginName;

class Plugin {
    // Implementation
}
```

### Prefixing (CRITICAL - prevents conflicts)
- All functions: `{prefix}` prefix
- All classes: `{prefix_}` prefix or namespace
- All hooks: `{prefix}` prefix
- All options: `{prefix}` prefix
- All database tables: `$wpdb->prefix . '{prefix}table'`

### File Organization
```
{slug}/
├── {slug}.php           # Main plugin file
├── composer.json       # Dependencies & autoload
├── phpunit.xml        # Test configuration
├── phpcs.xml          # Code standards
├── .editorconfig      # Editor settings
├── README.md          # GitHub readme
├── readme.txt         # WordPress.org readme
├── src/               # Source code (PSR-4)
│   ├── Core/         # Core functionality
│   ├── Admin/        # Admin-only code
│   ├── Public/       # Frontend code
│   ├── Database/     # Database handlers
│   └── REST/         # REST API endpoints
├── tests/             # Test files
├── admin/             # Admin assets
│   ├── css/
│   └── js/
├── public/            # Public assets
│   ├── css/
│   └── js/
├── includes/          # Shared includes (legacy)
├── languages/         # Translation files
│   ├── {slug}.pot
│   ├── {slug}-en_US.po
│   └── {slug}-en_US.mo
└── build/             # Build artifacts (gitignored)
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # SECURITY RULES
    # ═══════════════════════════════════════════════════════════════════

    SECURITY_RULES = """
## WordPress Security Rules (MANDATORY)

### Input Sanitization
```php
// ❌ NEVER trust user input
$user_input = $_POST['field'];

// ✅ ALWAYS sanitize
$user_input = sanitize_text_field( $_POST['field'] );
$user_email = sanitize_email( $_POST['email'] );
$user_url   = esc_url_raw( $_POST['url'] );
$user_int   = absint( $_POST['number'] );
$html       = wp_kses_post( $_POST['content'] );
```

### Output Escaping
```php
// ❌ NEVER output raw
echo $variable;

// ✅ ALWAYS escape
esc_html_e( $text, '{text_domain}' );           // Plain text
echo esc_html( $variable );                    // HTML entities
echo esc_attr( $variable );                    // HTML attributes
echo esc_url( $url );                          // URLs
echo esc_js( $javascript );                    // JavaScript
echo wp_kses_post( $html );                   // Allowed HTML
```

### Nonce Verification (CRITICAL)
```php
// Form generation
wp_nonce_field( '{prefix}action', '{prefix}nonce' );

// Form processing
if ( ! isset( $_POST['{prefix}nonce'] ) ||
     ! wp_verify_nonce( $_POST['{prefix}nonce'], '{prefix}action' ) ) {
    wp_die( esc_html__( 'Security check failed', '{text_domain}' ) );
}
```

### Capability Checks
```php
// ❌ NEVER assume admin
if ( is_admin() ) {  // WRONG!

// ✅ ALWAYS check capabilities
if ( ! current_user_can( 'manage_options' ) ) {
    wp_die( esc_html__( 'Insufficient permissions', '{text_domain}' ) );
}

// Common capabilities:
// - manage_options      (admin)
// - edit_posts          (editor)
// - publish_posts       (author)
// - edit_posts          (contributor)
// - read                (subscriber)
```

### Database Security
```php
// ❌ NEVER raw SQL
$wpdb->query( "SELECT * FROM $table WHERE id = $id" );

// ✅ ALWAYS use prepare()
$wpdb->query( $wpdb->prepare(
    "SELECT * FROM {{$wpdb->prefix}}{slug}_table WHERE id = %d AND status = %s",
    $id,
    $status
) );

// For INSERT/UPDATE, use data arrays
$wpdb->insert(
    $wpdb->prefix . '{slug}_table',
    array(
        'name'   => $name,
        'value'  => $value,
        'status' => 'active',
    ),
    array('%s', '%s', '%s')  // Format specifiers
);
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # BEST PRACTICES
    # ═══════════════════════════════════════════════════════════════════

    BEST_PRACTICES = """
## WordPress Plugin Best Practices

### Main Plugin File Structure
```php
<?php
/**
 * Plugin Name: {plugin_name}
 * Plugin URI:  https://example.com/{slug}
 * Description: Brief plugin description
 * Version:     {version}
 * Author:      {author}
 * License:     {license}
 * License URI: http://www.gnu.org/licenses/gpl-2.0.txt
 * Text Domain: {text_domain}
 * Domain Path: /languages
 * Requires at least: 5.8
 * Requires PHP: 7.4
 */

// Prevent direct access
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// Define constants
define( '{prefix}VERSION', '{version}' );
define( '{prefix}PLUGIN_DIR', plugin_dir_path( __FILE__ ) );
define( '{prefix}PLUGIN_URL', plugin_dir_url( __FILE__ ) );
define( '{prefix}PLUGIN_BASENAME', plugin_basename( __FILE__ ) );

// Autoloader (Composer or custom)
require_once {prefix}PLUGIN_DIR . 'vendor/autoload.php';

// Activation/Deactivation
register_activation_hook( __FILE__, '{prefix}activate' );
register_deactivation_hook( __FILE__, '{prefix}deactivate' );
register_uninstall_hook( __FILE__, '{prefix}uninstall' );

// Initialize
function {prefix}init() {
    $plugin = new \\{namespace}\\Core\\Plugin();
    $plugin->run();
}
add_action( 'plugins_loaded', '{prefix}init' );
```

### Assets Enqueuing
```php
// Admin assets
add_action( 'admin_enqueue_scripts', '{prefix}admin_assets' );
function {prefix}admin_assets( $hook ) {
    // Only load on plugin page
    if ( 'toplevel_page_{slug}' !== $hook ) {
        return;
    }

    wp_enqueue_style(
        '{slug}-admin',
        {prefix}PLUGIN_URL . 'admin/css/{slug}-admin.css',
        array(),
        {prefix}VERSION
    );

    wp_enqueue_script(
        '{slug}-admin',
        {prefix}PLUGIN_URL . 'admin/js/{slug}-admin.js',
        array( 'jquery' ),
        {prefix}VERSION,
        true  // In footer
    );

    // Localize script for AJAX
    wp_localize_script( '{slug}-admin', '{prefix}ajax', array(
        'ajax_url' => admin_url( 'admin-ajax.php' ),
        'nonce'    => wp_create_nonce( '{prefix}nonce' ),
    ) );
}
```

### i18n (Internationalization)
```php
// Load text domain
add_action( 'plugins_loaded', '{prefix}load_textdomain' );
function {prefix}load_textdomain() {
    load_plugin_textdomain(
        '{text_domain}',
        false,
        dirname( {prefix}PLUGIN_BASENAME ) . '/languages/'
    );
}

// Use translation functions
_e( 'Hello World', '{text_domain}' );                    // Echo
__( 'Hello World', '{text_domain}' );                    // Return
esc_html_e( 'Hello World', '{text_domain}' );            // Echo + escape
esc_html__( 'Hello World', '{text_domain}' );            // Return + escape
```
"""

    # ═══════════════════════════════════════════════════════════════════
    # CHECKLIST
    # ═══════════════════════════════════════════════════════════════════

    CHECKLIST = """
## WordPress Plugin Development Checklist

### Phase 1: Setup (10 min)
- [ ] Create Git repository
- [ ] Add .gitignore (wp-content, vendor, node_modules, build)
- [ ] Create composer.json with PSR-4 autoloading
- [ ] Add .editorconfig (WordPress style)
- [ ] Configure PHPCS with WordPress standards
- [ ] Setup phpunit.xml for testing
- [ ] Create README.md (GitHub)
- [ ] Create readme.txt (WordPress.org format)

### Phase 2: Structure (15 min)
- [ ] Create main plugin file with proper header
- [ ] Define plugin constants
- [ ] Setup autoloader (Composer or SPL)
- [ ] Create folder structure (src/, tests/, admin/, public/)
- [ ] Implement activation hook
- [ ] Implement deactivation hook
- [ ] Create uninstall.php
- [ ] Add text domain loading

### Phase 3: Core Development (20 min)
- [ ] Create main plugin class
- [ ] Implement dependency injection container
- [ ] Create service classes (single responsibility)
- [ ] Register WordPress hooks
- [ ] Implement admin menu/pages
- [ ] Create frontend functionality
- [ ] Add database handlers (if needed)

### Phase 4: Security (15 min) ⚠️ CRITICAL
- [ ] Nonce verification on all forms
- [ ] Capability checks before actions
- [ ] Sanitize all inputs with `sanitize_*()`
- [ ] Escape all outputs with `esc_*()`
- [ ] Use $wpdb->prepare() for SQL
- [ ] Add REST API permission callbacks
- [ ] Review for XSS vulnerabilities
- [ ] Check for CSRF protection

### Phase 5: Testing (15 min)
- [ ] Write unit tests with PHPUnit
- [ ] Setup wp-env for integration tests
- [ ] Test activation/deactivation
- [ ] Test uninstall cleanup
- [ ] Run PHPCS code standards check
- [ ] Test with WP_DEBUG enabled
- [ ] Test on clean WordPress install
- [ ] Test multisite compatibility (if applicable)

### Phase 6: Release Preparation (5 min)
- [ ] Update version number
- [ ] Write changelog
- [ ] Tag release in Git
- [ ] Run final security scan
- [ ] Test on WordPress.org standards
"""

    def __init__(self):
        """Initialize rules engine."""
        pass

    def get_architecture_path(self, path_key: str) -> dict[str, Any]:
        """Get architecture path details."""
        return self.ARCHITECTURE_PATHS.get(path_key, {})

    def get_all_architecture_paths(self) -> dict[str, dict[str, Any]]:
        """Get all available architecture paths."""
        return self.ARCHITECTURE_PATHS

    def recommend_architecture_path(
        self,
        public_distribution: bool = False,
        team_size: int = 1,
        complexity: str = "medium",
        timeline: str = "normal",
    ) -> str:
        """
        Recommend architecture path based on requirements.
        """
        if public_distribution or team_size > 1 or complexity == "complex":
            return "modular_oop"
        elif timeline == "tight" and complexity == "simple":
            return "lightweight"
        elif complexity == "complex" and not public_distribution:
            return "headless"

        return "modular_oop"

    def generate_config(
        self, plugin_name: str, architecture_path: str | None = None, **kwargs
    ) -> WPRulesConfig:
        """
        Generate WordPress plugin configuration.
        """
        # Generate slug from name
        slug = plugin_name.lower().replace(" ", "-").replace("_", "-")

        # Generate namespace
        namespace_parts = [p.capitalize() for p in slug.split("-")]
        namespace = "\\\\".join(namespace_parts)

        # Generate prefix
        prefix = slug.replace("-", "_") + "_"

        # Determine architecture path
        if architecture_path is None:
            architecture_path = self.recommend_architecture_path(
                public_distribution=kwargs.get("public_distribution", True),
                team_size=kwargs.get("team_size", 1),
                complexity=kwargs.get("complexity", "medium"),
            )

        return WPRulesConfig(
            plugin_name=plugin_name,
            plugin_slug=slug,
            text_domain=slug,
            namespace=namespace,
            prefix=prefix,
            version=kwargs.get("version", "1.0.0"),
            author=kwargs.get("author", ""),
            license=kwargs.get("license", "GPL-2.0+"),
            architecture_path=architecture_path,
            include_composer=kwargs.get("include_composer", True),
            include_tests=kwargs.get("include_tests", True),
            include_ci=kwargs.get("include_ci", True),
            include_i18n=kwargs.get("include_i18n", True),
            include_rest=kwargs.get("include_rest", False),
            include_blocks=kwargs.get("include_blocks", False),
            strict_security=kwargs.get("strict_security", True),
        )

    def get_rules_file_content(self, config: WPRulesConfig) -> str:
        """Generate .ai-rules.md content for WordPress plugin."""
        path_info = self.get_architecture_path(config.architecture_path)

        # Replace placeholders
        coding_standards = self.CODING_STANDARDS.format(
            slug=config.plugin_slug,
            prefix=config.prefix,
            text_domain=config.text_domain,
        )

        security_rules = self.SECURITY_RULES.format(
            prefix=config.prefix,
            text_domain=config.text_domain,
            slug=config.plugin_slug,
        )

        best_practices = self.BEST_PRACTICES.format(
            plugin_name=config.plugin_name,
            slug=config.plugin_slug,
            prefix=config.prefix,
            namespace=config.namespace,
            text_domain=config.text_domain,
            version=config.version,
            author=config.author,
            license=config.license,
        )

        content = f"""# WordPress Plugin Rules: {config.plugin_name}

## 🎯 Architecture Decision
**Path Selected**: {path_info.get('name', config.architecture_path)}

**Description**: {path_info.get('description', '')}

**Pros**:
{chr(10).join(['- ' + p for p in path_info.get('pros', [])])}

**Cons**:
{chr(10).join(['- ' + c for c in path_info.get('cons', [])])}

## 📋 Plugin Configuration
- **Plugin Name**: {config.plugin_name}
- **Plugin Slug**: `{config.plugin_slug}`
- **Text Domain**: `{config.text_domain}`
- **Namespace**: `{config.namespace}`
- **Prefix**: `{config.prefix}`
- **Version**: {config.version}
- **License**: {config.license}

## 🚨 MUST FOLLOW RULES

### 1. Prefix Everything
ALL functions, classes, hooks, and options must use prefix: `{config.prefix}`

### 2. Security (NON-NEGOTIABLE)
- ✅ Sanitize ALL inputs with `sanitize_*()` functions
- ✅ Escape ALL outputs with `esc_*()` functions
- ✅ Use nonces for form processing
- ✅ Check capabilities with `current_user_can()`
- ✅ Use `$wpdb->prepare()` for all SQL queries

### 3. Coding Standards
- ✅ Use TABS for indentation (not spaces)
- ✅ Follow WordPress PHP Coding Standards
- ✅ All functions must be prefixed: `{config.prefix}function_name()`
- ✅ Use namespace: `{config.namespace}` for classes

### 4. Text Domain
Use text domain `{config.text_domain}` for ALL strings:
```php
__( 'Hello', '{config.text_domain}' );
```

## 📚 Generated Rules

{coding_standards}

{security_rules}

{best_practices}

## ✅ Development Checklist

{self.CHECKLIST}

## 📖 References
- [WordPress Plugin Handbook](https://developer.wordpress.org/plugins/)
- [WordPress Coding Standards](https://developer.wordpress.org/coding-standards/)
- [WordPress Security](https://developer.wordpress.org/plugins/security/)
- [Plugin Readme Standards](https://wordpress.org/plugins/readme.txt)
"""
        return content

    def save_rules_file(self, config: WPRulesConfig, output_dir: Path) -> Path:
        """Save rules file to output directory."""
        rules_content = self.get_rules_file_content(config)
        rules_file = output_dir / ".ai-rules.md"
        rules_file.write_text(rules_content, encoding="utf-8")
        return rules_file


# Convenience function
def generate_wordpress_plugin_rules(plugin_name: str, output_dir: Path, **kwargs) -> Path:
    """
    Generate WordPress plugin rules file.

    Args:
        plugin_name: Name of the plugin
        output_dir: Directory to save rules file
        **kwargs: Additional configuration options

    Returns:
        Path to generated rules file
    """
    rules = WordPressPluginRules()
    config = rules.generate_config(plugin_name, **kwargs)
    return rules.save_rules_file(config, output_dir)


if __name__ == "__main__":
    # Demo
    rules = WordPressPluginRules()

    print("=" * 70)
    print("WordPress Plugin Development Rules")
    print("=" * 70)

    print("\n📚 Architecture Paths:")
    for key, path in rules.get_all_architecture_paths().items():
        print(f"\n  {path['name']} ({key})")
        print(f"    {path['description'][:80]}...")

    print("\n\n🎯 Recommendation Example:")
    recommended = rules.recommend_architecture_path(
        public_distribution=True, team_size=3, complexity="complex"
    )
    print(f"  Recommended: {recommended}")

    print("\n\n📄 Generate Rules for 'My Awesome Plugin':")
    config = rules.generate_config("My Awesome Plugin")
    print(f"  Slug: {config.plugin_slug}")
    print(f"  Namespace: {config.namespace}")
    print(f"  Prefix: {config.prefix}")
    print(f"  Architecture: {config.architecture_path}")

    print("\n\n✅ WordPress Plugin Rules Engine Ready!")
