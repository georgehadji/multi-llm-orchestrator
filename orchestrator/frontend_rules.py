"""
Front-End Web Development Rules
================================
Production-ready rules for modern front-end web applications.

Stack: React + TypeScript + Vite/Next.js + Tailwind + Zustand + TanStack Query
Architecture: Feature-based, scalable, strict typing

Usage:
    from orchestrator.frontend_rules import FrontendRules

    rules = FrontendRules()
    config = rules.generate_config("My SaaS", template="saas")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FrontendConfig:
    """Front-end project configuration."""
    project_name: str
    project_slug: str

    # Template choice
    template: str = "saas"  # saas, dashboard, ai_first, minimal, microfrontend

    # Core stack
    framework: str = "react"  # react, vue, svelte
    language: str = "typescript"
    bundler: str = "vite"  # vite, nextjs, webpack

    # State management
    server_state: str = "tanstack_query"  # tanstack_query, swr, rtk_query
    client_state: str = "zustand"  # zustand, redux_toolkit, jotai

    # UI
    styling: str = "tailwind"  # tailwind, antd, material, chakra
    component_library: bool = True

    # Forms
    form_library: str = "react_hook_form"  # react_hook_form, formik
    validation: str = "zod"  # zod, yup, joi

    # Testing
    test_framework: str = "vitest"  # vitest, jest
    e2e_framework: str = "playwright"  # playwright, cypress
    coverage_threshold: float = 80.0

    # Quality
    strict_typescript: bool = True
    eslint_strict: bool = True
    prettier: bool = True

    # CI/CD
    ci_provider: str = "github_actions"  # github_actions, gitlab_ci
    docker: bool = True

    # Features
    ssr: bool = False
    pwa: bool = False
    i18n: bool = False
    analytics: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "template": self.template,
            "stack": f"{self.framework}+{self.language}",
        }


class FrontendRules:
    """
    Front-end Web Development Rules Engine.

    Provides production-ready templates and best practices
    for modern React + TypeScript applications.
    """

    # ═══════════════════════════════════════════════════════════════════
    # TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    TEMPLATES = {
        "saas": {
            "name": "SaaS Application",
            "description": "Full-featured SaaS with auth, billing, teams, and admin panels.",
            "recommended_for": [
                "SaaS products",
                "Subscription services",
                "Multi-tenant applications",
                "B2B platforms",
            ],
            "features": [
                "Authentication (OAuth, SSO)",
                "Team/Organization support",
                "Billing integration (Stripe)",
                "Role-based permissions",
                "Admin dashboard",
                "User onboarding",
                "Settings panels",
            ],
            "stack_addons": [
                "Stripe integration",
                "Clerk/Auth0 for auth",
                "React Query for caching",
                "Recharts for analytics",
            ],
            "structure": [
                "features/auth/",
                "features/billing/",
                "features/teams/",
                "features/admin/",
                "features/settings/",
                "features/dashboard/",
            ],
        },
        "dashboard": {
            "name": "Data Dashboard",
            "description": "High-performance dashboard for data visualization and analytics.",
            "recommended_for": [
                "Analytics platforms",
                "Monitoring tools",
                "Business intelligence",
                "Data-heavy applications",
            ],
            "features": [
                "Real-time data updates",
                "Interactive charts",
                "Data tables with filtering",
                "Export functionality",
                "Drill-down navigation",
                "Custom date ranges",
                "Dashboard widgets",
            ],
            "stack_addons": [
                "TanStack Table",
                "Recharts/D3",
                "WebSockets/SSE",
                "Virtualization for large lists",
            ],
            "structure": [
                "features/analytics/",
                "features/reports/",
                "features/widgets/",
                "features/data-explorer/",
            ],
        },
        "ai_first": {
            "name": "AI-First Application",
            "description": "LLM-powered app with streaming, chat UI, and token management.",
            "recommended_for": [
                "ChatGPT clones",
                "AI assistants",
                "Content generation tools",
                "LLM-integrated apps",
            ],
            "features": [
                "Streaming responses (SSE)",
                "Chat interface",
                "Message history",
                "Token counting",
                "Prompt templates",
                "Model selection",
                "File uploads (RAG)",
            ],
            "stack_addons": [
                "SSE for streaming",
                "Markdown rendering",
                "Syntax highlighting",
                "File upload (S3)",
            ],
            "structure": [
                "features/chat/",
                "features/prompts/",
                "features/models/",
                "features/history/",
            ],
        },
        "minimal": {
            "name": "Lean Startup",
            "description": "Ultra-minimal setup for rapid prototyping and MVPs.",
            "recommended_for": [
                "Rapid prototyping",
                "MVPs",
                "Landing pages",
                "Proof of concepts",
            ],
            "features": [
                "Minimal dependencies",
                "Fast build times",
                "Simple state management",
                "Basic routing",
            ],
            "stack_addons": [],
            "structure": [
                "features/landing/",
                "features/app/",
            ],
            "notes": "Remove any non-essential packages. Focus on speed.",
        },
        "microfrontend": {
            "name": "Micro-Frontend Architecture",
            "description": "Scalable architecture with independent deployable modules.",
            "recommended_for": [
                "Large enterprise apps",
                "Multi-team projects",
                "Independent deployments",
                "Module federation",
            ],
            "features": [
                "Module federation",
                "Independent builds",
                "Shared component library",
                "Runtime integration",
            ],
            "stack_addons": [
                "Webpack Module Federation",
                "Single-spa (optional)",
                "Shared dependencies",
            ],
            "structure": [
                "apps/shell/",
                "apps/auth/",
                "apps/dashboard/",
                "apps/settings/",
                "packages/ui/",
                "packages/shared/",
            ],
        },
    }

    # ═══════════════════════════════════════════════════════════════════
    # CORE RULES
    # ═══════════════════════════════════════════════════════════════════

    ARCHITECTURE_RULES = """
## Feature-Based Architecture

### Folder Structure
```
project-name/
├── src/
│   ├── main.tsx              # Entry point
│   ├── App.tsx               # Root component
│   ├── routes/
│   │   └── router.tsx        # Route definitions
│   │
│   ├── app/                  # App-level configuration
│   │   ├── providers.tsx     # Context providers
│   │   ├── store.ts          # Global store setup
│   │   └── queryClient.ts    # React Query client
│   │
│   ├── shared/               # Reusable across features
│   │   ├── components/       # Generic components
│   │   ├── ui/              # Design system wrappers
│   │   ├── hooks/           # Shared hooks
│   │   ├── utils/           # Utilities
│   │   ├── types/           # Global types
│   │   └── constants/       # Constants
│   │
│   ├── features/             # Feature-based modules
│   │   ├── auth/
│   │   │   ├── api.ts       # API calls
│   │   │   ├── hooks.ts     # Feature hooks
│   │   │   ├── store.ts     # Feature state
│   │   │   ├── types.ts     # Feature types
│   │   │   ├── components/  # Feature components
│   │   │   └── pages/       # Feature pages
│   │   │
│   │   └── dashboard/
│   │       ├── api.ts
│   │       ├── hooks.ts
│   │       ├── components/
│   │       └── pages/
│   │
│   └── styles/
│       └── global.css
│
├── tests/
│   ├── unit/
│   └── e2e/
│
├── public/
│   └── favicon.svg
│
└── scripts/
    └── build.sh
```

### Feature Isolation Principles

Each feature MUST contain:
- **api.ts** - All API calls for the feature
- **hooks.ts** - Feature-specific React hooks
- **store.ts** - Feature state (if needed)
- **types.ts** - TypeScript types
- **components/** - Feature-specific components
- **pages/** - Route-level components

```typescript
// features/auth/api.ts
import { apiClient } from '@/shared/lib/api';

export const authApi = {
  login: (credentials: LoginCredentials) =>
    apiClient.post('/auth/login', credentials),

  logout: () =>
    apiClient.post('/auth/logout'),

  getMe: () =>
    apiClient.get('/auth/me'),
};

// features/auth/hooks.ts
import { useQuery, useMutation } from '@tanstack/react-query';
import { authApi } from './api';

export const useAuth = () => {
  return useQuery({
    queryKey: ['auth', 'me'],
    queryFn: authApi.getMe,
  });
};

export const useLogin = () => {
  return useMutation({
    mutationFn: authApi.login,
  });
};

// features/auth/components/LoginForm.tsx
import { useLogin } from '../hooks';

export const LoginForm = () => {
  const login = useLogin();
  // Component implementation
};
```

### Import Rules

```typescript
// ✅ ABSOLUTE imports with path aliases
import { Button } from '@/shared/ui/Button';
import { useAuth } from '@/features/auth/hooks';

// ❌ NO relative imports across features
import { useAuth } from '../../../auth/hooks'; // WRONG!

// ✅ Relative imports within feature only
import { LoginForm } from './LoginForm'; // OK
```
"""

    TYPESCRIPT_RULES = """
## TypeScript Strict Mode (REQUIRED)

### tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/features/*": ["./src/features/*"],
      "@/shared/*": ["./src/shared/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Type Safety Rules

```typescript
// ✅ ALWAYS define return types for functions
export const fetchUser = async (id: string): Promise<User> => {
  // Implementation
};

// ✅ NEVER use 'any'
const data: any = response; // ❌ FORBIDDEN!

// ✅ Use 'unknown' and type guard
const data: unknown = response;
if (isUser(data)) {
  // Now data is typed as User
}

// ✅ Zod schemas for runtime validation
import { z } from 'zod';

const UserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  name: z.string().min(1),
});

type User = z.infer<typeof UserSchema>;

// Validate API responses
const response = await fetchUser(id);
const user = UserSchema.parse(response); // Runtime check
```
"""

    STATE_MANAGEMENT = """
## State Management Strategy

### Server State (TanStack Query)

```typescript
// app/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 3,
      refetchOnWindowFocus: false,
    },
  },
});

// features/users/api.ts
export const usersApi = {
  getAll: () => fetch('/api/users').then(r => r.json()),
  getById: (id: string) => fetch(`/api/users/${id}`).then(r => r.json()),
};

// features/users/hooks.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export const useUsers = () => {
  return useQuery({
    queryKey: ['users'],
    queryFn: usersApi.getAll,
  });
};

export const useUser = (id: string) => {
  return useQuery({
    queryKey: ['users', id],
    queryFn: () => usersApi.getById(id),
    enabled: !!id,
  });
};

export const useCreateUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: usersApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
};
```

### Client State (Zustand)

```typescript
// features/auth/store.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        isAuthenticated: false,
        setUser: (user) => set({ user, isAuthenticated: !!user }),
        logout: () => set({ user: null, isAuthenticated: false }),
      }),
      { name: 'auth-storage' }
    )
  )
);
```

### Form State (React Hook Form + Zod)

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const loginSchema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Password too short'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export const LoginForm = () => {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = (data: LoginFormData) => {
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('email')} />
      {errors.email && <span>{errors.email.message}</span>}

      <input type="password" {...register('password')} />
      {errors.password && <span>{errors.password.message}</span>}

      <button type="submit">Login</button>
    </form>
  );
};
```
"""

    TESTING_RULES = """
## Testing Strategy

### Unit Tests (Vitest)

```typescript
// features/auth/hooks.test.ts
import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuth } from './hooks';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useAuth', () => {
  it('should fetch user data', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toBeDefined();
  });
});
```

### Component Tests

```typescript
// shared/components/Button.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);

    fireEvent.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when loading', () => {
    render(<Button isLoading>Loading</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### E2E Tests (Playwright)

```typescript
// tests/e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('user can login', async ({ page }) => {
    await page.goto('/login');

    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('text=Welcome')).toBeVisible();
  });

  test('shows error on invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.fill('[name="email"]', 'invalid@example.com');
    await page.fill('[name="password"]', 'wrong');
    await page.click('button[type="submit"]');

    await expect(page.locator('text=Invalid credentials')).toBeVisible();
  });
});
```

### Test Coverage

```json
// vitest.config.ts
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
      exclude: [
        'node_modules/',
        'tests/',
        '*.config.*',
      ],
    },
  },
});
```
"""

    CI_CD_RULES = """
## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Type check
        run: npm run type-check

      - name: Lint
        run: npm run lint

      - name: Format check
        run: npm run format:check

  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    runs-on: ubuntu-latest
    needs: [quality, test]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Upload build artifact
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

  e2e:
    runs-on: ubuntu-latest
    needs: build

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: playwright-report
          path: playwright-report/
```

### Docker (Production)

```dockerfile
# Dockerfile
# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Stage 2: Production
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost/ || exit 1
```

```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /assets {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```
"""

    SECURITY_RULES = """
## Security Baseline

### Content Security Policy

```typescript
// index.html
<meta http-equiv="Content-Security-Policy"
  content="
    default-src 'self';
    script-src 'self' 'unsafe-inline';
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    connect-src 'self' https://api.example.com;
  ">
```

### XSS Prevention

```typescript
// ✅ NEVER use dangerouslySetInnerHTML
<div dangerouslySetInnerHTML={{ __html: content }} /> // ❌ FORBIDDEN!

// ✅ Use safe rendering
import DOMPurify from 'dompurify';

const SafeContent = ({ html }: { html: string }) => {
  const clean = DOMPurify.sanitize(html);
  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
};

// ✅ For markdown, use safe libraries
import ReactMarkdown from 'react-markdown';

<ReactMarkdown>{markdownContent}</ReactMarkdown>
```

### Authentication

```typescript
// ✅ Store tokens in httpOnly cookies (server handles)
// ❌ NEVER store sensitive data in localStorage

// Token refresh logic
import { QueryClient } from '@tanstack/react-query';

const queryClient = new QueryClient({
  queryCache: {
    onError: (error) => {
      if (error.response?.status === 401) {
        // Redirect to login
        window.location.href = '/login';
      }
    },
  },
});
```

### Environment Variables

```typescript
// .env.example
VITE_API_URL=https://api.example.com
VITE_APP_ENV=development
VITE_SENTRY_DSN=

// ✅ Public variables only (VITE_ prefix)
const API_URL = import.meta.env.VITE_API_URL;

// ❌ NEVER expose secrets
const SECRET_KEY = import.meta.env.VITE_SECRET_KEY; // WRONG!
```
"""

    PERFORMANCE_RULES = """
## Performance Optimization

### Code Splitting

```typescript
// ✅ Lazy load routes
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./features/dashboard/pages/Dashboard'));
const Settings = lazy(() => import('./features/settings/pages/Settings'));

// Router configuration
{
  path: '/dashboard',
  element: (
    <Suspense fallback={<PageLoader />}>
      <Dashboard />
    </Suspense>
  ),
}

// ✅ Lazy load heavy components
const HeavyChart = lazy(() => import('./HeavyChart'));
```

### Bundle Size Monitoring

```typescript
// vite.config.ts
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      open: true,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
        },
      },
    },
    chunkSizeWarningLimit: 500,
  },
});
```

### Image Optimization

```typescript
// ✅ Use WebP with fallbacks
<picture>
  <source srcSet="image.webp" type="image/webp" />
  <source srcSet="image.jpg" type="image/jpeg" />
  <img src="image.jpg" alt="Description" loading="lazy" />
</picture>

// ✅ Lazy loading
<img src="image.jpg" loading="lazy" alt="Description" />

// ✅ Proper sizing
<img
  src="image-800w.jpg"
  srcSet="image-400w.jpg 400w, image-800w.jpg 800w"
  sizes="(max-width: 600px) 400px, 800px"
  alt="Description"
/>
```
"""

    def __init__(self):
        """Initialize rules engine."""
        pass

    def get_template(self, template_key: str) -> dict[str, Any]:
        """Get template details."""
        return self.TEMPLATES.get(template_key, {})

    def get_all_templates(self) -> dict[str, dict[str, Any]]:
        """Get all available templates."""
        return self.TEMPLATES

    def recommend_template(self,
                          project_type: str = "saas",
                          team_size: int = 1,
                          complexity: str = "medium") -> str:
        """Recommend template based on requirements."""
        recommendations = {
            "saas": "saas",
            "dashboard": "dashboard",
            "analytics": "dashboard",
            "ai": "ai_first",
            "chatbot": "ai_first",
            "mvp": "minimal",
            "landing": "minimal",
            "enterprise": "microfrontend",
            "large": "microfrontend",
        }

        return recommendations.get(project_type, "saas")

    def generate_config(self,
                       project_name: str,
                       template: str | None = None,
                       **kwargs) -> FrontendConfig:
        """Generate front-end project configuration."""
        slug = project_name.lower().replace(' ', '-').replace('_', '-')

        if template is None:
            template = self.recommend_template(
                project_type=kwargs.get('project_type', 'saas'),
                team_size=kwargs.get('team_size', 1),
            )

        return FrontendConfig(
            project_name=project_name,
            project_slug=slug,
            template=template,
            framework=kwargs.get('framework', 'react'),
            language=kwargs.get('language', 'typescript'),
            bundler=kwargs.get('bundler', 'vite'),
            server_state=kwargs.get('server_state', 'tanstack_query'),
            client_state=kwargs.get('client_state', 'zustand'),
            styling=kwargs.get('styling', 'tailwind'),
            component_library=kwargs.get('component_library', True),
            form_library=kwargs.get('form_library', 'react_hook_form'),
            validation=kwargs.get('validation', 'zod'),
            test_framework=kwargs.get('test_framework', 'vitest'),
            e2e_framework=kwargs.get('e2e_framework', 'playwright'),
            coverage_threshold=kwargs.get('coverage_threshold', 80.0),
            strict_typescript=kwargs.get('strict_typescript', True),
            eslint_strict=kwargs.get('eslint_strict', True),
            prettier=kwargs.get('prettier', True),
            ci_provider=kwargs.get('ci_provider', 'github_actions'),
            docker=kwargs.get('docker', True),
            ssr=kwargs.get('ssr', False),
            pwa=kwargs.get('pwa', False),
            i18n=kwargs.get('i18n', False),
            analytics=kwargs.get('analytics', False),
        )

    def get_rules_file_content(self, config: FrontendConfig) -> str:
        """Generate .ai-rules.md content for front-end project."""
        template_info = self.get_template(config.template)

        content = f"""# Front-End Development Rules: {config.project_name}

## 🎯 Project Configuration

**Template**: {template_info.get('name', config.template)}
**Stack**: {config.framework.title()} + {config.language.title()} + {config.bundler.title()}
**State**: {config.server_state.replace('_', ' ').title()} + {config.client_state.replace('_', ' ').title()}
**Styling**: {config.styling.title()}

### Template Description
{template_info.get('description', '')}

### Included Features
{chr(10).join(['- ' + f for f in template_info.get('features', [])])}

## 📋 Technology Stack

### Core
- **Framework**: {config.framework} with {config.language}
- **Bundler**: {config.bundler}
- **Server State**: {config.server_state}
- **Client State**: {config.client_state}

### UI
- **Styling**: {config.styling}
- **Component Library**: {'Yes' if config.component_library else 'No'}

### Forms
- **Library**: {config.form_library}
- **Validation**: {config.validation}

### Testing
- **Unit**: {config.test_framework}
- **E2E**: {config.e2e_framework}
- **Coverage**: {config.coverage_threshold}%

### Quality
- **TypeScript Strict**: {'Yes' if config.strict_typescript else 'No'}
- **ESLint Strict**: {'Yes' if config.eslint_strict else 'No'}
- **Prettier**: {'Yes' if config.prettier else 'No'}

### DevOps
- **CI/CD**: {config.ci_provider}
- **Docker**: {'Yes' if config.docker else 'No'}

## 🚨 MUST FOLLOW RULES

### 1. Feature-Based Architecture
✅ Organize code by features, not by type
- Each feature contains: api.ts, hooks.ts, components/, pages/
- Shared code in shared/
- No cross-feature imports

### 2. TypeScript Strict Mode
✅ Enable all strict options
- No `any` types
- Explicit return types
- Zod schemas for validation

### 3. State Management
✅ Separate server and client state
- Server state: TanStack Query
- Client state: Zustand
- No duplication

### 4. Testing
✅ Minimum {config.coverage_threshold}% coverage
- Unit tests for logic
- Component tests for UI
- E2E for critical flows

### 5. Performance
✅ Optimize for production
- Code splitting
- Lazy loading
- Bundle monitoring

{self.ARCHITECTURE_RULES}

{self.TYPESCRIPT_RULES}

{self.STATE_MANAGEMENT}

{self.TESTING_RULES}

{self.CI_CD_RULES}

{self.SECURITY_RULES}

{self.PERFORMANCE_RULES}

## ✅ Production Readiness Checklist

### Code Quality
- [ ] TypeScript strict mode enabled
- [ ] No ESLint errors
- [ ] No `console.log` in production
- [ ] Proper error boundaries
- [ ] Loading states implemented

### Testing
- [ ] Unit tests > {config.coverage_threshold}% coverage
- [ ] Component tests for shared components
- [ ] E2E tests for critical flows
- [ ] All tests passing in CI

### Performance
- [ ] Code splitting implemented
- [ ] Lazy loading for routes
- [ ] Images optimized
- [ ] Bundle size < 500KB (initial)
- [ ] Lighthouse score > 90

### Security
- [ ] CSP headers configured
- [ ] No secrets in client code
- [ ] XSS prevention
- [ ] Input validation

### Deployment
- [ ] Docker image builds
- [ ] Environment variables configured
- [ ] Health check endpoint
- [ ] CDN configured for assets

## 📖 References

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TanStack Query](https://tanstack.com/query/latest)
- [Zustand](https://docs.pmnd.rs/zustand/getting-started/introduction)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Vitest](https://vitest.dev/)
- [Playwright](https://playwright.dev/)
"""
        return content

    def save_rules_file(self, config: FrontendConfig, output_dir: Path) -> Path:
        """Save rules file to output directory."""
        rules_content = self.get_rules_file_content(config)
        rules_file = output_dir / ".ai-rules.md"
        rules_file.write_text(rules_content, encoding="utf-8")
        return rules_file


# Convenience function
def generate_frontend_rules(
    project_name: str,
    output_dir: Path,
    **kwargs
) -> Path:
    """Generate front-end rules file."""
    rules = FrontendRules()
    config = rules.generate_config(project_name, **kwargs)
    return rules.save_rules_file(config, output_dir)


if __name__ == "__main__":
    # Demo
    rules = FrontendRules()

    print("=" * 70)
    print("Front-End Development Rules")
    print("=" * 70)

    print("\n📚 Templates:")
    for key, template in rules.get_all_templates().items():
        print(f"\n  {template['name']} ({key})")
        print(f"    {template['description'][:60]}...")

    print("\n\n🎯 Generate Rules for 'SaaS Dashboard':")
    config = rules.generate_config("SaaS Dashboard", template="dashboard")
    print(f"  Project: {config.project_name}")
    print(f"  Slug: {config.project_slug}")
    print(f"  Template: {config.template}")
    print(f"  Stack: {config.framework} + {config.language}")
    print(f"  State: {config.server_state} + {config.client_state}")

    print("\n\n✅ Front-End Rules Engine Ready!")
