"""ContactForm component — generated with rate-limit error handling."""

CONTACT_FORM_TEMPLATE = """import { useState } from 'react'

/**
 * ContactForm - Rate-limited contact form component.
 *
 * SECURITY: This form posts to a server-side API route that MUST implement
 * IP-based rate limiting. The client handles 429 (Too Many Requests) with a
 * user-visible error. Do NOT embed API keys in this file.
 */
export default function ContactForm({
  headline = '{headline}',
}: {
  headline?: string
}) {
  const [formState, setFormState] = useState<'idle' | 'sending' | 'sent' | 'error' | 'rate-limited'>('idle')
  const [errorMessage, setErrorMessage] = useState('')

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const form = e.currentTarget
    const data = new FormData(form)

    setFormState('sending')
    setErrorMessage('')

    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: data.get('name'),
          email: data.get('email'),
          message: data.get('message'),
        }),
      })

      if (res.status === 429) {
        setFormState('rate-limited')
        setErrorMessage('Too many requests. Please wait a moment and try again.')
        return
      }

      if (!res.ok) throw new Error('Server error')

      setFormState('sent')
      form.reset()
    } catch {
      setFormState('error')
      setErrorMessage('Something went wrong. Please try again later.')
    }
  }

  return (
    <section id="contact" className="w-full max-w-2xl mx-auto px-4 py-16" style={{ fontFamily: '{font_body}' }}>
      <h2 className="text-3xl font-bold text-center mb-8">{headline}</h2>

      {formState === 'sent' ? (
        <div className="p-6 rounded-lg text-center" style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }}>
          <p className="text-lg font-medium">Thank you for reaching out!</p>
          <p className="text-sm mt-2 opacity-70">We will respond within 24 hours.</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-6" noValidate>
          <div>
            <label htmlFor="contact-name" className="block text-sm font-medium mb-1">Name</label>
            <input id="contact-name" name="name" type="text" required
              className="w-full px-4 py-2.5 rounded-lg transition-colors focus:outline-none focus:ring-2"
              style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }} />
          </div>
          <div>
            <label htmlFor="contact-email" className="block text-sm font-medium mb-1">Email</label>
            <input id="contact-email" name="email" type="email" required
              className="w-full px-4 py-2.5 rounded-lg transition-colors focus:outline-none focus:ring-2"
              style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }} />
          </div>
          <div>
            <label htmlFor="contact-message" className="block text-sm font-medium mb-1">Message</label>
            <textarea id="contact-message" name="message" required rows={5}
              className="w-full px-4 py-2.5 rounded-lg resize-y transition-colors focus:outline-none focus:ring-2"
              style={{ backgroundColor: '{surface_alt}', borderColor: '{border}' }} />
          </div>

          {formState === 'rate-limited' && (
            <div role="alert" className="p-3 rounded-lg text-sm font-medium"
              style={{ backgroundColor: '#fef2f2', borderColor: '#fecaca', color: '#dc2626' }}>
              {errorMessage}
            </div>
          )}

          {formState === 'error' && (
            <div role="alert" className="p-3 rounded-lg text-sm opacity-70"
              style={{ backgroundColor: '{surface_alt}' }}>
              {errorMessage}
            </div>
          )}

          <button type="submit" disabled={formState === 'sending' || formState === 'rate-limited'}
            className="w-full py-3 px-6 rounded-lg font-medium transition-all duration-200 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50"
            style={{ backgroundColor: '{primary}', color: '#ffffff' }}>
            {formState === 'sending' ? 'Sending...' : 'Send Message'}
          </button>

          <p className="text-xs text-center opacity-60">
            Your data is handled securely. We never share your information.
          </p>
        </form>
      )}
    </section>
  )
}

/*
 * API Route (Next.js): /app/api/contact/route.ts
 * This route MUST include IP-based rate limiting:
 *
 *   import { Ratelimit } from "@upstash/ratelimit"
 *   import { Redis } from "@upstash/redis"
 *
 *   const ratelimit = new Ratelimit({
 *     redis: Redis.fromEnv(),
 *     limiter: Ratelimit.slidingWindow(3, "1 h"),
 *   })
 *
 *   export async function POST(req: Request) {
 *     const ip = req.headers.get("x-forwarded-for") ?? "unknown"
 *     const { success } = await ratelimit.limit(ip)
 *     if (!success) {
 *       return Response.json({ error: "Too many requests" }, { status: 429 })
 *     }
 *     const body = await req.json()
 *     // ... process contact form
 *     return Response.json({ success: true })
 *   }
 */
"""
