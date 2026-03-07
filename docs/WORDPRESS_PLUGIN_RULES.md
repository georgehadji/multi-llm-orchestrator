# WordPress Plugin Development Rules

Comprehensive rules engine for professional WordPress plugin development.

## 🎯 Architecture Paths

### 1. Modular OOP + Composer + Tests (Recommended)
**For**: Public distribution, team collaboration, long-term maintenance

```
my-plugin/
├── composer.json       # PSR-4 autoloading
├── src/
│   ├── Core/          # Core classes
│   ├── Admin/         # Admin functionality
│   └── Public/        # Frontend code
├── tests/             # PHPUnit + wp-env
├── .github/workflows/ # CI/CD
└── phpcs.xml          # WordPress standards
```

**Pros**:
- ✅ High maintainability
- ✅ Scalable architecture
- ✅ PSR-4 autoloading
- ✅ Full test coverage
- ✅ CI/CD ready

### 2. Lightweight Procedural
**For**: Small internal projects, MVP, rapid prototyping

```
my-plugin/
├── my-plugin.php      # Main file
├── includes/          # Functions
├── admin/            # Admin screens
└── public/           # Frontend
```

**Pros**:
- ✅ Fast delivery
- ✅ No build process
- ✅ Simple to understand

### 3. Headless / Service-Oriented
**For**: SaaS integration, heavy compute, multi-platform

```
my-plugin/
├── src/
│   ├── ApiClient/    # External service client
│   ├── Cache/        # Response caching
│   └── Auth/         # Authentication
```

**Pros**:
- ✅ Clean separation
- ✅ Scalable backend
- ✅ Multi-platform ready

## 🚀 Quick Start

### Generate Rules for Your Plugin

```python
from orchestrator.wordpress_plugin_rules import (
    WordPressPluginRules,
    generate_wordpress_plugin_rules
)

# Method 1: Using the function
rules_file = generate_wordpress_plugin_rules(
    plugin_name="My Awesome Plugin",
    output_dir=Path("./output"),
    public_distribution=True,
    team_size=3,
    complexity="complex"
)

# Method 2: Using the class
rules = WordPressPluginRules()
config = rules.generate_config(
    plugin_name="My Awesome Plugin",
    architecture_path="modular_oop",  # or auto-detect
    version="1.0.0",
    author="Your Name",
    include_tests=True,
    include_ci=True,
)
rules.save_rules_file(config, Path("./output"))
```

### Auto-Detect Architecture Path

```python
# Automatically recommends best path
config = rules.generate_config(
    plugin_name="My Plugin",
    public_distribution=True,  # WP.org → modular_oop
    team_size=3,              # Team → modular_oop
    complexity="complex",      # Complex → modular_oop
    timeline="normal"
)
```

## 📋 Generated Rules Include

### 1. Security Rules (MANDATORY)
```php
// ❌ NEVER do this
$user_input = $_POST['field'];
echo $variable;
$wpdb->query("SELECT * FROM table WHERE id = $id");

// ✅ ALWAYS do this
$user_input = sanitize_text_field($_POST['field']);
echo esc_html($variable);
$wpdb->query($wpdb->prepare("SELECT * FROM table WHERE id = %d", $id));
```

### 2. Coding Standards
- Use TABS (not spaces)
- Prefix everything
- WordPress naming conventions
- Namespacing for OOP

### 3. File Structure
- Main plugin file with proper header
- Composer autoloading (PSR-4)
- Separation of concerns
- Asset organization

### 4. Best Practices
- Activation/Deactivation/Uninstall hooks
- Proper asset enqueuing
- i18n implementation
- Options API usage

## 🛡️ Security Checklist

- [ ] Nonce verification on all forms
- [ ] Capability checks with `current_user_can()`
- [ ] Input sanitization with `sanitize_*()`
- [ ] Output escaping with `esc_*()`
- [ ] SQL prepared statements
- [ ] REST API permission callbacks
- [ ] XSS prevention
- [ ] CSRF protection

## 🧪 Testing Requirements

```bash
# Code standards
./vendor/bin/phpcs --standard=WordPress

# Static analysis
./vendor/bin/phpstan analyse

# Unit tests
./vendor/bin/phpunit

# Integration tests
npm run test:e2e
```

## 📦 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        php: ['7.4', '8.0', '8.1', '8.2']
    
    steps:
    - uses: actions/checkout@v3
    - name: Setup PHP
      uses: shivammathur/setup-php@v2
      with:
        php-version: ${{ matrix.php }}
    - name: Install dependencies
      run: composer install
    - name: Run PHPCS
      run: ./vendor/bin/phpcs --standard=WordPress
    - name: Run tests
      run: ./vendor/bin/phpunit
```

## 📖 Example Output

Generated `.ai-rules.md` file:

```markdown
# WordPress Plugin Rules: My Awesome Plugin

## 🎯 Architecture Decision
**Path Selected**: Modular OOP + Composer + Tests

## 📋 Plugin Configuration
- **Plugin Name**: My Awesome Plugin
- **Plugin Slug**: `my-awesome-plugin`
- **Text Domain**: `my-awesome-plugin`
- **Namespace**: `MyAwesomePlugin`
- **Prefix**: `my_awesome_plugin_`

## 🚨 MUST FOLLOW RULES

### 1. Prefix Everything
ALL functions must use prefix: `my_awesome_plugin_`

### 2. Security (NON-NEGOTIABLE)
- ✅ Sanitize ALL inputs
- ✅ Escape ALL outputs
- ✅ Use nonces
- ✅ Check capabilities

### 3. Coding Standards
- ✅ Use TABS
- ✅ Follow WordPress standards
- ✅ Namespace: `MyAwesomePlugin`

## 📚 Generated Rules
[Full coding standards...]
[Security rules...]
[Best practices...]

## ✅ Development Checklist
[Full checklist...]
```

## 🎨 Architecture Decision Matrix

| Factor | Modular OOP | Lightweight | Headless |
|--------|-------------|-------------|----------|
| Public Distribution | ✅ Best | ⚠️ Risky | ⚠️ Complex |
| Team Size > 1 | ✅ Best | ❌ Poor | ✅ Good |
| Complex Logic | ✅ Best | ❌ Poor | ✅ Good |
| Rapid MVP | ⚠️ Slow | ✅ Fast | ❌ Slow |
| Long-term Maint | ✅ Best | ❌ Poor | ✅ Good |
| Learning Curve | ⚠️ Steep | ✅ Easy | ⚠️ Steep |

## 🔧 Integration with Orchestrator

```python
from orchestrator import Orchestrator, Budget
from orchestrator.wordpress_plugin_rules import WordPressPluginRules

# Create rules
rules = WordPressPluginRules()
config = rules.generate_config("WooCommerce Enhancer")

# Generate rules file
rules.save_rules_file(config, Path("./output"))

# Use with orchestrator
orch = Orchestrator(budget=Budget(max_usd=10.0))
state = await orch.run_project(
    project_description="Build WooCommerce enhancement plugin",
    success_criteria="Follow all WordPress coding standards",
    output_dir=Path("./output")
)
```

## 📚 References

- [WordPress Plugin Handbook](https://developer.wordpress.org/plugins/)
- [WordPress Coding Standards](https://developer.wordpress.org/coding-standards/)
- [WordPress Security](https://developer.wordpress.org/plugins/security/)
- [Plugin Readme Standards](https://wordpress.org/plugins/readme.txt)
- [PHPCS WordPress](https://github.com/WordPress/WordPress-Coding-Standards)

## ✅ Summary

| Feature | Status |
|---------|--------|
| 3 Architecture Paths | ✅ |
| Security Rules | ✅ |
| Coding Standards | ✅ |
| Best Practices | ✅ |
| Development Checklist | ✅ |
| Auto-detection | ✅ |
| CI/CD Config | ✅ |
