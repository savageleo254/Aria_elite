import { NextRequest, NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE || 'http://127.0.0.1:8000'
const SUPERVISOR_TOKEN = process.env.SUPERVISOR_API_TOKEN || process.env.NEXT_PUBLIC_SUPERVISOR_TOKEN || 'dev_token'

export async function GET(_request: NextRequest) {
  // Return empty signals until backend provides a signals listing endpoint
  return NextResponse.json({ signals: [] })
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const res = await fetch(`${BACKEND_BASE}/signals/generate`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SUPERVISOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return NextResponse.json({ error: 'Backend /signals/generate failed', detail: text }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend /signals/generate' }, { status: 502 })
  }
}
