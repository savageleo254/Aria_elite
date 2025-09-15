import { NextRequest, NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE || 'http://127.0.0.1:8000'
const SUPERVISOR_TOKEN = process.env.SUPERVISOR_API_TOKEN || process.env.NEXT_PUBLIC_SUPERVISOR_TOKEN || 'dev_token'

export async function POST(request: NextRequest, { params }: { params: { agent: string } }) {
  try {
    const agentName = params.agent
    
    const res = await fetch(`${BACKEND_BASE}/agents/${agentName}/disable`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SUPERVISOR_TOKEN}`,
        'Content-Type': 'application/json'
      }
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return NextResponse.json({ error: `Backend /agents/${agentName}/disable failed`, detail: text }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend agent disable endpoint' }, { status: 502 })
  }
}
