import { NextRequest, NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE || 'http://127.0.0.1:8000'
const SUPERVISOR_TOKEN = process.env.SUPERVISOR_API_TOKEN || process.env.NEXT_PUBLIC_SUPERVISOR_TOKEN || 'dev_token'

export async function GET(_request: NextRequest) {
  try {
    const res = await fetch(`${BACKEND_BASE}/config/risk`, {
      headers: {
        'Authorization': `Bearer ${SUPERVISOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      cache: 'no-store'
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return NextResponse.json({ error: 'Backend /config/risk failed', detail: text }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend /config/risk' }, { status: 502 })
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    
    const res = await fetch(`${BACKEND_BASE}/config/risk`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${SUPERVISOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return NextResponse.json({ error: 'Backend /config/risk update failed', detail: text }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend /config/risk' }, { status: 502 })
  }
}
