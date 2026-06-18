"""Auth page component — generated with email verification flow."""

AUTH_TEMPLATE = """import { useState } from 'react'

/**
 * {component_name} - {page_type} page.
 *
 * SECURITY: Registration sends email verification token before enabling login.
 * API routes are server-side only. No API keys or secrets in this file.
 */
export default function {component_name}({{
  headline = '{headline}',
}}: {{
  headline?: string
}}) {{
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [state, setState] = useState<'idle' | 'loading' | 'success' | 'check-email' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {{
    e.preventDefault()
    setState('loading')
    setErrorMessage('')

    try {{
      const endpoint = '{endpoint}'
      const res = await fetch(endpoint, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ email, password }}),
      }})

      if (res.status === 429) {{
        setState('error')
        setErrorMessage('Too many attempts. Please wait and try again.')
        return
      }}

      if (!res.ok) {{
        const data = await res.json().catch(() => ({{}}))
        throw new Error(data.error || 'Request failed')
      }}

      {success_state}
    }} catch (err) {{
      setState('error')
      setErrorMessage(err instanceof Error ? err.message : 'Something went wrong')
    }}
  }}

  return (
    <section className="w-full max-w-md mx-auto px-4 py-16" style={{ fontFamily: '{font_body}' }}>
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">{page_title}</h1>
        <p className="opacity-70">{page_subtitle}</p>
      </div>

      {state === 'check-email' ? (
        <div className="p-6 rounded-lg text-center" style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }}>
          <h2 className="text-xl font-semibold mb-3">Check your email</h2>
          <p className="mb-4">
            We sent a verification link to <strong>{email}</strong>.
            Please check your inbox and click the link to activate your account.
          </p>
          <p className="text-sm opacity-60">
            Did not receive the email? Check your spam folder or{' '}
            <button type="button" className="underline hover:opacity-80" style={{ color: '{primary}' }}>
              resend verification
            </button>
          </p>
        </div>
      ) : state === 'success' ? (
        <div className="p-6 rounded-lg text-center" style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }}>
          <p className="text-lg font-medium">{success_message}</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-5" noValidate>
          <div>
            <label htmlFor="auth-email" className="block text-sm font-medium mb-1">Email</label>
            <input id="auth-email" type="email" required value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg transition-colors focus:outline-none focus:ring-2"
              style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }} />
          </div>
          <div>
            <label htmlFor="auth-password" className="block text-sm font-medium mb-1">Password</label>
            <input id="auth-password" type="password" required value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={8}
              className="w-full px-4 py-2.5 rounded-lg transition-colors focus:outline-none focus:ring-2"
              style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }} />
          </div>

          {state === 'error' && (
            <div role="alert" className="p-3 rounded-lg text-sm font-medium"
              style={{ backgroundColor: '#fef2f2', borderColor: '#fecaca', color: '#dc2626' }}>
              {errorMessage}
            </div>
          )}

          <button type="submit" disabled={state === 'loading'}
            className="w-full py-3 px-6 rounded-lg font-medium transition-all duration-200 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50"
            style={{ backgroundColor: '{primary}', color: '#ffffff' }}>
            {state === 'loading' ? 'Please wait...' : '{button_text}'}
          </button>

          {verify_note}
        </form>
      )}

      <p className="text-center mt-6 text-sm opacity-70">
        {switch_message}
      </p>
    </section>
  )
}}

/*
 * API Routes (Next.js):
 *
 * /app/api/auth/register/route.ts - Registration with IP rate limiting:
 *   const ratelimit = new Ratelimit({{ redis: Redis.fromEnv(), limiter: Ratelimit.slidingWindow(5, "1 h") }})
 *   export async function POST(req: Request) {{
 *     const ip = req.headers.get("x-forwarded-for") ?? "unknown"
 *     const {{ success }} = await ratelimit.limit(ip)
 *     if (!success) return Response.json({{ error: "Rate limited" }}, {{ status: 429 }})
 *     const {{ email, password }} = await req.json()
 *     const token = crypto.randomUUID()
 *     // Save user with verified=false, store token
 *     // Send verification email: /api/auth/verify-email?token={{token}}
 *     return Response.json({{ message: "Verification email sent" }})
 *   }}
 *
 * /app/api/auth/verify-email/route.ts - Email verification:
 *   export async function GET(req: Request) {{
 *     const token = new URL(req.url).searchParams.get("token")
 *     // Mark user as verified=true
 *     return Response.redirect("/login?verified=true")
 *   }}
 */
"""
