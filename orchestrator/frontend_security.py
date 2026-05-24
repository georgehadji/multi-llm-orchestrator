"""
Front-End Security Components — Component Pattern + Functional Components
==========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Front-end security component generation using Component Pattern for React/Vue/Angular
and Functional Components for pure, testable utilities.

Paradigm: Hybrid (OOP for complex components, Functional for simple utilities)
Patterns: Component Pattern, Functional Components, Factory Method, Builder

Usage:
    from orchestrator.frontend_security import generate_csp_meta_tag, create_csrf_component

    csp_tag = generate_csp_meta_tag(policy=default_csp_policy)
    csrf_component = create_csrf_component()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CSPDirective:
    """
    Immutable CSP directive.

    Attributes:
        name: Directive name (e.g., "default-src", "script-src")
        values: Directive values (e.g., ["'self'", "https://cdn.example.com"])
        report_only: If True, report violations without blocking
    """

    name: str
    values: List[str] = field(default_factory=list)
    report_only: bool = False

    def to_string(self) -> str:
        """Convert to CSP string format."""
        values_str = " ".join(self.values) if self.values else ""
        return f"{self.name} {values_str}".strip()


@dataclass(frozen=True)
class CSPolicy:
    """
    Immutable Content Security Policy.

    Attributes:
        directives: List of CSP directives
        report_uri: URI for violation reports
        report_only: If True, use Content-Security-Policy-Report-Only header
    """

    directives: List[CSPDirective] = field(default_factory=list)
    report_uri: Optional[str] = None
    report_only: bool = False

    def to_header(self) -> str:
        """Convert to HTTP header value."""
        parts = [d.to_string() for d in self.directives]

        if self.report_uri:
            parts.append(f"report-uri {self.report_uri}")

        return "; ".join(parts)

    def to_meta_tag(self) -> str:
        """Convert to HTML meta tag."""
        header_name = (
            "Content-Security-Policy-Report-Only" if self.report_only else "Content-Security-Policy"
        )
        content = self.to_header()
        return f'<meta http-equiv="{header_name}" content="{content}">'


@dataclass(frozen=True)
class CSRFConfig:
    """
    Immutable CSRF protection configuration.

    Attributes:
        token_name: Form field name for CSRF token
        header_name: Header name for AJAX requests
        cookie_name: Cookie name for token storage
        secure: Require HTTPS
        same_site: SameSite cookie attribute
    """

    token_name: str = "csrf_token"
    header_name: str = "X-CSRF-Token"
    cookie_name: str = "csrf_token"
    secure: bool = True
    same_site: str = "Strict"


# ═══════════════════════════════════════════════════════════════════
# FUNCTIONAL COMPONENTS (Pure Functions)
# ═══════════════════════════════════════════════════════════════════

# Type alias for functional components
ReactFunctionalComponent = Callable[[Dict[str, Any]], str]
VueFunctionalComponent = Callable[[Dict[str, Any]], str]


def generate_csp_meta_tag(
    default_src: List[str] = None,
    script_src: List[str] = None,
    style_src: List[str] = None,
    img_src: List[str] = None,
    connect_src: List[str] = None,
    font_src: List[str] = None,
    object_src: List[str] = None,
    media_src: List[str] = None,
    frame_src: List[str] = None,
    base_uri: List[str] = None,
    form_action: List[str] = None,
    frame_ancestors: List[str] = None,
    upgrade_insecure_requests: bool = False,
    report_uri: str = None,
    report_only: bool = False,
) -> str:
    """
    Pure function: Generate CSP meta tag.

    Functional Programming: No side effects, deterministic output.

    Args:
        default_src: Default source list
        script_src: Script source list
        style_src: Style source list
        img_src: Image source list
        connect_src: Connection source list (AJAX, WebSocket)
        font_src: Font source list
        object_src: Object/embed source list
        media_src: Media source list
        frame_src: Frame/iframe source list
        base_uri: Base URI source list
        form_action: Form action URLs
        frame_ancestors: Allowed frame ancestors
        upgrade_insecure_requests: Upgrade HTTP to HTTPS
        report_uri: Violation report URI
        report_only: Use report-only mode

    Returns:
        HTML meta tag string

    Example:
        >>> csp = generate_csp_meta_tag(
        ...     default_src=["'self'"],
        ...     script_src=["'self'", "https://cdn.example.com"],
        ...     style_src=["'self'", "'unsafe-inline'"]
        ... )
        >>> print(csp)
        <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdn.example.com; ...">
    """
    directives = []

    # Add directives with values
    if default_src:
        directives.append(CSPDirective(name="default-src", values=default_src))
    if script_src:
        directives.append(CSPDirective(name="script-src", values=script_src))
    if style_src:
        directives.append(CSPDirective(name="style-src", values=style_src))
    if img_src:
        directives.append(CSPDirective(name="img-src", values=img_src))
    if connect_src:
        directives.append(CSPDirective(name="connect-src", values=connect_src))
    if font_src:
        directives.append(CSPDirective(name="font-src", values=font_src))
    if object_src:
        directives.append(CSPDirective(name="object-src", values=object_src))
    if media_src:
        directives.append(CSPDirective(name="media-src", values=media_src))
    if frame_src:
        directives.append(CSPDirective(name="frame-src", values=frame_src))
    if base_uri:
        directives.append(CSPDirective(name="base-uri", values=base_uri))
    if form_action:
        directives.append(CSPDirective(name="form-action", values=form_action))
    if frame_ancestors:
        directives.append(CSPDirective(name="frame-ancestors", values=frame_ancestors))

    # Special directives
    if upgrade_insecure_requests:
        directives.append(CSPDirective(name="upgrade-insecure-requests"))

    # Build policy
    policy = CSPolicy(directives=directives, report_uri=report_uri, report_only=report_only)

    return policy.to_meta_tag()


def generate_csrf_token() -> str:
    """
    Pure function: Generate CSRF token.

    Returns:
        Cryptographically secure CSRF token
    """
    import secrets

    return secrets.token_hex(32)


def create_csrf_input_component(token: str, config: CSRFConfig = None) -> str:
    """
    Pure function: Create CSRF token input component.

    Args:
        token: CSRF token value
        config: CSRF configuration

    Returns:
        HTML input element string
    """
    if config is None:
        config = CSRFConfig()

    return f'<input type="hidden" name="{config.token_name}" value="{token}">'


def create_csrf_meta_component(token: str, config: CSRFConfig = None) -> str:
    """
    Pure function: Create CSRF token meta component.

    Args:
        token: CSRF token value
        config: CSRF configuration

    Returns:
        HTML meta element string
    """
    if config is None:
        config = CSRFConfig()

    return f'<meta name="csrf-token" content="{token}">'


def escape_html(value: str) -> str:
    """
    Pure function: Escape HTML special characters.

    Functional Programming: Pure function for XSS prevention.

    Args:
        value: String to escape

    Returns:
        Escaped string

    Example:
        >>> escape_html("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;'
    """
    if not isinstance(value, str):
        return str(value)

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def sanitize_user_input(value: str, allowed_tags: List[str] = None) -> str:
    """
    Pure function: Sanitize user input (basic XSS prevention).

    Note: For production, use a library like DOMPurify or bleach.
    This is a basic implementation for demonstration.

    Args:
        value: User input to sanitize
        allowed_tags: List of allowed HTML tags

    Returns:
        Sanitized string
    """
    if not value:
        return ""

    # Remove script tags and event handlers
    import re

    # Remove script tags
    value = re.sub(
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", value, flags=re.IGNORECASE
    )

    # Remove event handlers
    value = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', "", value, flags=re.IGNORECASE)
    value = re.sub(r"on\w+\s*=\s*[^\s>]+", "", value, flags=re.IGNORECASE)

    # Remove javascript: protocol
    value = re.sub(r"javascript:", "", value, flags=re.IGNORECASE)

    # If no allowed tags, escape everything
    if not allowed_tags:
        return escape_html(value)

    return value


# ═══════════════════════════════════════════════════════════════════
# FUNCTIONAL COMPONENT FACTORIES
# ═══════════════════════════════════════════════════════════════════


def create_secure_form_component(
    action: str,
    method: str = "POST",
    csrf_token: str = None,
    csrf_config: CSRFConfig = None,
    enctype: str = None,
    class_name: str = "secure-form",
) -> ReactFunctionalComponent:
    """
    Factory: Create secure form functional component.

    Args:
        action: Form action URL
        method: HTTP method
        csrf_token: CSRF token
        csrf_config: CSRF configuration
        enctype: Form encoding type
        class_name: CSS class name

    Returns:
        React functional component
    """

    def component(props: Dict[str, Any]) -> str:
        """Secure form component."""
        csrf_cfg = csrf_config or CSRFConfig()
        token = csrf_token or generate_csrf_token()

        # Build form attributes
        attrs = [
            f'action="{escape_html(action)}"',
            f'method="{method.upper()}"',
            f'class="{class_name}"',
        ]

        if enctype:
            attrs.append(f'enctype="{enctype}"')

        # Build form HTML
        html = f'<form {" ".join(attrs)}>'

        # Add CSRF token
        html += create_csrf_input_component(token, csrf_cfg)

        # Add children from props
        children = props.get("children", "")
        html += children

        html += "</form>"

        return html

    return component


def create_password_input_component(
    name: str = "password",
    min_length: int = 8,
    require_uppercase: bool = True,
    require_lowercase: bool = True,
    require_digit: bool = True,
    require_special: bool = True,
    show_strength_meter: bool = True,
) -> ReactFunctionalComponent:
    """
    Factory: Create password input with validation component.

    Args:
        name: Input name
        min_length: Minimum password length
        require_uppercase: Require uppercase letters
        require_lowercase: Require lowercase letters
        require_digit: Require digits
        require_special: Require special characters
        show_strength_meter: Show password strength meter

    Returns:
        React functional component
    """

    def component(props: Dict[str, Any]) -> str:
        """Password input component."""
        # Build validation rules
        pattern = ""
        if require_uppercase:
            pattern += "(?=.*[A-Z])"
        if require_lowercase:
            pattern += "(?=.*[a-z])"
        if require_digit:
            pattern += "(?=.*\\d)"
        if require_special:
            pattern += '(?=.*[!@#$%^&*(),.?":{}|<>])'
        pattern += f".{{{min_length},}}"

        # Build HTML
        html = f"""
<div class="password-input-container">
  <label for="{name}">Password</label>
  <input
    type="password"
    id="{name}"
    name="{name}"
    minlength="{min_length}"
    pattern="{pattern}"
    required
    aria-describedby="password-requirements"
  />
  <div id="password-requirements" class="password-requirements">
    <ul>
      <li>Minimum {min_length} characters</li>
      {f'<li>At least one uppercase letter</li>' if require_uppercase else ''}
      {f'<li>At least one lowercase letter</li>' if require_lowercase else ''}
      {f'<li>At least one number</li>' if require_digit else ''}
      {f'<li>At least one special character</li>' if require_special else ''}
    </ul>
  </div>
  {f'<div id="password-strength-meter" class="strength-meter"></div>' if show_strength_meter else ''}
</div>
"""
        return html

    return component


def create_auth_component(
    component_type: str = "login",
    provider: str = "jwt",
    include_2fa: bool = False,
) -> ReactFunctionalComponent:
    """
    Factory: Create authentication component.

    Args:
        component_type: "login", "register", or "forgot_password"
        provider: Authentication provider ("jwt", "session", "oauth")
        include_2fa: Include 2FA support

    Returns:
        React functional component
    """

    def component(props: Dict[str, Any]) -> str:
        """Authentication component."""
        if component_type == "login":
            return _render_login_form(provider, include_2fa)
        elif component_type == "register":
            return _render_register_form(provider)
        else:
            return _render_forgot_password_form()

    return component


def _render_login_form(provider: str, include_2fa: bool) -> str:
    """Render login form."""
    two_fa_html = ""
    if include_2fa:
        two_fa_html = """
  <div id="two-factor-container" style="display: none;">
    <label for="two-factor-code">Two-Factor Authentication Code</label>
    <input
      type="text"
      id="two-factor-code"
      name="two_factor_code"
      pattern="[0-9]{6}"
      maxlength="6"
      placeholder="Enter 6-digit code"
    />
  </div>
"""

    return f"""
<form class="login-form" action="/api/auth/login" method="POST">
  {create_csrf_input_component(generate_csrf_token())}
  
  <div class="form-group">
    <label for="email">Email</label>
    <input
      type="email"
      id="email"
      name="email"
      required
      autocomplete="email"
    />
  </div>
  
  <div class="form-group">
    <label for="password">Password</label>
    <input
      type="password"
      id="password"
      name="password"
      required
      autocomplete="current-password"
    />
  </div>
  
  {two_fa_html}
  
  <div class="form-group">
    <button type="submit" class="btn btn-primary">Login</button>
  </div>
  
  <div class="form-footer">
    <a href="/forgot-password">Forgot password?</a>
    <a href="/register">Create account</a>
  </div>
</form>
"""


def _render_register_form(provider: str) -> str:
    """Render registration form."""
    return f"""
<form class="register-form" action="/api/auth/register" method="POST">
  {create_csrf_input_component(generate_csrf_token())}
  
  <div class="form-group">
    <label for="name">Full Name</label>
    <input
      type="text"
      id="name"
      name="name"
      required
      autocomplete="name"
    />
  </div>
  
  <div class="form-group">
    <label for="email">Email</label>
    <input
      type="email"
      id="email"
      name="email"
      required
      autocomplete="email"
    />
  </div>
  
  <div class="form-group">
    <label for="password">Password</label>
    <input
      type="password"
      id="password"
      name="password"
      required
      autocomplete="new-password"
      minlength="8"
    />
    <div class="password-requirements">
      <small>Minimum 8 characters, include uppercase, lowercase, number, and special character</small>
    </div>
  </div>
  
  <div class="form-group">
    <label for="password-confirm">Confirm Password</label>
    <input
      type="password"
      id="password-confirm"
      name="password_confirm"
      required
      autocomplete="new-password"
    />
  </div>
  
  <div class="form-group">
    <label class="checkbox-label">
      <input type="checkbox" name="terms" required />
      I agree to the <a href="/terms" target="_blank">Terms of Service</a>
      and <a href="/privacy" target="_blank">Privacy Policy</a>
    </label>
  </div>
  
  <div class="form-group">
    <button type="submit" class="btn btn-primary">Create Account</button>
  </div>
  
  <div class="form-footer">
    Already have an account? <a href="/login">Login</a>
  </div>
</form>
"""


def _render_forgot_password_form() -> str:
    """Render forgot password form."""
    return f"""
<form class="forgot-password-form" action="/api/auth/forgot-password" method="POST">
  {create_csrf_input_component(generate_csrf_token())}
  
  <div class="form-group">
    <label for="email">Email</label>
    <input
      type="email"
      id="email"
      name="email"
      required
      autocomplete="email"
      placeholder="Enter your email address"
    />
  </div>
  
  <div class="form-group">
    <button type="submit" class="btn btn-primary">Send Reset Link</button>
  </div>
  
  <div class="form-footer">
    <a href="/login">Back to login</a>
  </div>
</form>
"""


# ═══════════════════════════════════════════════════════════════════
# OOP COMPONENTS (Complex Stateful Components)
# ═══════════════════════════════════════════════════════════════════


class SecurityComponent(ABC):
    """
    Abstract base class for security components.

    Component Pattern: Common interface for all security components.
    """

    @abstractmethod
    def render(self) -> str:
        """
        Render component to HTML string.

        Returns:
            HTML string
        """
        pass

    @abstractmethod
    def get_props(self) -> Dict[str, Any]:
        """
        Get component props.

        Returns:
            Props dictionary
        """
        pass


@dataclass
class CSPMetaTagComponent(SecurityComponent):
    """
    OOP Component: CSP meta tag.

    Attributes:
        policy: CSP policy
        nonce: Optional nonce for inline scripts
    """

    policy: CSPolicy
    nonce: Optional[str] = None

    def render(self) -> str:
        """Render CSP meta tag."""
        return self.policy.to_meta_tag()

    def get_props(self) -> Dict[str, Any]:
        """Get component props."""
        return {
            "policy": self.policy,
            "nonce": self.nonce,
        }


@dataclass
class CSRFTokenComponent(SecurityComponent):
    """
    OOP Component: CSRF token input.

    Attributes:
        token: CSRF token
        config: CSRF configuration
        render_as: "input" or "meta"
    """

    token: str
    config: CSRFConfig = field(default_factory=CSRFConfig)
    render_as: str = "input"

    def render(self) -> str:
        """Render CSRF component."""
        if self.render_as == "meta":
            return create_csrf_meta_component(self.token, self.config)
        else:
            return create_csrf_input_component(self.token, self.config)

    def get_props(self) -> Dict[str, Any]:
        """Get component props."""
        return {
            "token": self.token,
            "config": self.config,
            "render_as": self.render_as,
        }


@dataclass
class SecureFormComponent(SecurityComponent):
    """
    OOP Component: Secure form with CSRF protection.

    Attributes:
        action: Form action URL
        method: HTTP method
        csrf_token: CSRF token
        csrf_config: CSRF configuration
        enctype: Form encoding type
        class_name: CSS class name
        children: Form children HTML
        novalidate: Disable HTML5 validation
    """

    action: str
    method: str = "POST"
    csrf_token: str = field(default_factory=generate_csrf_token)
    csrf_config: CSRFConfig = field(default_factory=CSRFConfig)
    enctype: Optional[str] = None
    class_name: str = "secure-form"
    children: str = ""
    novalidate: bool = False

    def render(self) -> str:
        """Render secure form."""
        attrs = [
            f'action="{escape_html(self.action)}"',
            f'method="{self.method.upper()}"',
            f'class="{self.class_name}"',
        ]

        if self.enctype:
            attrs.append(f'enctype="{self.enctype}"')

        if self.novalidate:
            attrs.append("novalidate")

        html = f'<form {" ".join(attrs)}>'
        html += create_csrf_input_component(self.csrf_token, self.csrf_config)
        html += self.children
        html += "</form>"

        return html

    def get_props(self) -> Dict[str, Any]:
        """Get component props."""
        return {
            "action": self.action,
            "method": self.method,
            "csrf_token": self.csrf_token,
            "csrf_config": self.csrf_config,
            "enctype": self.enctype,
            "class_name": self.class_name,
            "children": self.children,
            "novalidate": self.novalidate,
        }

    # Fluent interface methods
    def with_csrf(self, token: str, config: CSRFConfig = None) -> "SecureFormComponent":
        """Set CSRF token (fluent interface)."""
        object.__setattr__(self, "csrf_token", token)
        if config:
            object.__setattr__(self, "csrf_config", config)
        return self

    def with_enctype(self, enctype: str) -> "SecureFormComponent":
        """Set encoding type (fluent interface)."""
        object.__setattr__(self, "enctype", enctype)
        return self

    def with_class(self, class_name: str) -> "SecureFormComponent":
        """Set CSS class (fluent interface)."""
        object.__setattr__(self, "class_name", class_name)
        return self

    def with_children(self, children: str) -> "SecureFormComponent":
        """Set children HTML (fluent interface)."""
        object.__setattr__(self, "children", children)
        return self


@dataclass
class PasswordStrengthMeterComponent(SecurityComponent):
    """
    OOP Component: Password strength meter.

    Attributes:
        target_element: Target password input ID
        show_feedback: Show textual feedback
        show_bar: Show visual strength bar
        min_length: Minimum length for valid password
    """

    target_element: str
    show_feedback: bool = True
    show_bar: bool = True
    min_length: int = 8

    def render(self) -> str:
        """Render password strength meter."""
        html = f'<div id="password-strength-meter" data-target="{self.target_element}">'

        if self.show_bar:
            html += """
  <div class="strength-bar">
    <div class="strength-fill" data-strength="0"></div>
  </div>
"""

        if self.show_feedback:
            html += """
  <div class="strength-feedback" data-feedback>
    Password strength: <span class="strength-text">Not entered</span>
  </div>
  <ul class="password-requirements">
    <li data-requirement="length">Minimum 8 characters</li>
    <li data-requirement="uppercase">At least one uppercase letter</li>
    <li data-requirement="lowercase">At least one lowercase letter</li>
    <li data-requirement="digit">At least one number</li>
    <li data-requirement="special">At least one special character</li>
  </ul>
"""

        html += """
<script>
(function() {
  const passwordInput = document.getElementById('%s');
  const strengthFill = document.querySelector('[data-strength]');
  const strengthText = document.querySelector('[data-feedback] .strength-text');
  const requirements = document.querySelectorAll('[data-requirement]');

  if (!passwordInput) return;

  passwordInput.addEventListener('input', function() {
    const password = this.value;
    let strength = 0;

    // Check requirements
    const hasLength = password.length >= %d;
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasDigit = /\\d/.test(password);
    const hasSpecial = /[!@#$%%^&*(),.?":{}|<>]/.test(password);

    // Update requirement indicators
    requirements.forEach(req => {
      const type = req.dataset.requirement;
      const met = {
        'length': hasLength,
        'uppercase': hasUppercase,
        'lowercase': hasLowercase,
        'digit': hasDigit,
        'special': hasSpecial
      }[type];

      req.classList.toggle('met', met);
      req.classList.toggle('not-met', !met);
    });

    // Calculate strength score
    if (hasLength) strength += 1;
    if (hasUppercase) strength += 1;
    if (hasLowercase) strength += 1;
    if (hasDigit) strength += 1;
    if (hasSpecial) strength += 1;

    // Update UI
    const percentage = (strength / 5) * 100;
    strengthFill.style.width = percentage + '%%';
    strengthFill.setAttribute('data-strength', strength);

    const labels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong', 'Very Strong'];
    const colors = ['#ff4444', '#ff8800', '#ffaa00', '#ffcc00', '#88cc00', '#44cc00'];

    if (strengthText) {
      strengthText.textContent = password ? labels[strength] : 'Not entered';
      strengthText.style.color = colors[strength];
    }
  });
})();
</script>
""" % (self.target_element, self.min_length)

        html += "</div>"

        return html

    def get_props(self) -> Dict[str, Any]:
        """Get component props."""
        return {
            "target_element": self.target_element,
            "show_feedback": self.show_feedback,
            "show_bar": self.show_bar,
            "min_length": self.min_length,
        }


# ═══════════════════════════════════════════════════════════════════
# COMPONENT FACTORY
# ═══════════════════════════════════════════════════════════════════


class SecurityComponentFactory:
    """
    Factory Method for security components.

    Usage:
        factory = SecurityComponentFactory()
        csp_component = factory.create_csp_component(policy)
        csrf_component = factory.create_csrf_component()
        form_component = factory.create_secure_form(action="/login")
    """

    def create_csp_component(
        self,
        policy: CSPolicy,
        nonce: str = None,
    ) -> CSPMetaTagComponent:
        """Create CSP meta tag component."""
        return CSPMetaTagComponent(policy=policy, nonce=nonce)

    def create_csrf_component(
        self,
        token: str = None,
        config: CSRFConfig = None,
        render_as: str = "input",
    ) -> CSRFTokenComponent:
        """Create CSRF token component."""
        return CSRFTokenComponent(
            token=token or generate_csrf_token(),
            config=config or CSRFConfig(),
            render_as=render_as,
        )

    def create_secure_form(
        self,
        action: str,
        method: str = "POST",
        csrf_token: str = None,
        **kwargs,
    ) -> SecureFormComponent:
        """Create secure form component."""
        return SecureFormComponent(
            action=action,
            method=method,
            csrf_token=csrf_token or generate_csrf_token(),
            **kwargs,
        )

    def create_password_meter(
        self,
        target_element: str,
        **kwargs,
    ) -> PasswordStrengthMeterComponent:
        """Create password strength meter component."""
        return PasswordStrengthMeterComponent(
            target_element=target_element,
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def generate_default_csp() -> CSPolicy:
    """
    Generate default CSP for web applications.

    Returns:
        Default CSP policy
    """
    return CSPolicy(
        directives=[
            CSPDirective(name="default-src", values=["'self'"]),
            CSPDirective(name="script-src", values=["'self'"]),
            CSPDirective(name="style-src", values=["'self'", "'unsafe-inline'"]),
            CSPDirective(name="img-src", values=["'self'", "data:", "https:"]),
            CSPDirective(name="font-src", values=["'self'", "https://fonts.gstatic.com"]),
            CSPDirective(name="connect-src", values=["'self'"]),
            CSPDirective(name="frame-ancestors", values=["'none'"]),
            CSPDirective(name="base-uri", values=["'self'"]),
            CSPDirective(name="form-action", values=["'self'"]),
        ],
        report_only=False,
    )


def generate_strict_csp() -> CSPolicy:
    """
    Generate strict CSP for high-security applications.

    Returns:
        Strict CSP policy
    """
    return CSPolicy(
        directives=[
            CSPDirective(name="default-src", values=["'none'"]),
            CSPDirective(name="script-src", values=["'self'"]),
            CSPDirective(name="style-src", values=["'self'"]),
            CSPDirective(name="img-src", values=["'self'"]),
            CSPDirective(name="font-src", values=["'self'"]),
            CSPDirective(name="connect-src", values=["'self'"]),
            CSPDirective(name="frame-ancestors", values=["'none'"]),
            CSPDirective(name="base-uri", values=["'none'"]),
            CSPDirective(name="form-action", values=["'self'"]),
            CSPDirective(name="object-src", values=["'none'"]),
        ],
        report_only=False,
    )


def generate_csp_for_react() -> CSPolicy:
    """
    Generate CSP optimized for React applications.

    Note: React requires 'unsafe-inline' for styles in development.
    For production, consider using nonce-based CSP.

    Returns:
        React-optimized CSP policy
    """
    return CSPolicy(
        directives=[
            CSPDirective(name="default-src", values=["'self'"]),
            CSPDirective(name="script-src", values=["'self'", "'unsafe-eval'"]),
            CSPDirective(name="style-src", values=["'self'", "'unsafe-inline'"]),
            CSPDirective(name="img-src", values=["'self'", "data:", "https:"]),
            CSPDirective(name="font-src", values=["'self'", "https://fonts.gstatic.com"]),
            CSPDirective(name="connect-src", values=["'self'", "https://api.example.com"]),
            CSPDirective(name="frame-ancestors", values=["'none'"]),
        ],
        report_only=False,
    )
