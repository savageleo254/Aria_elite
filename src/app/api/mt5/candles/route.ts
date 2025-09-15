import { NextRequest, NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE || 'http://127.0.0.1:8000'
const SUPERVISOR_TOKEN = process.env.SUPERVISOR_API_TOKEN || process.env.NEXT_PUBLIC_SUPERVISOR_TOKEN || 'dev_token'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get('symbol') || 'XAUUSD'
    const timeframe = searchParams.get('timeframe') || 'M5'
    const count = searchParams.get('count') || '100'

    const res = await fetch(`${BACKEND_BASE}/mt5/candles?symbol=${symbol}&timeframe=${timeframe}&count=${count}`, {
      headers: {
        'Authorization': `Bearer ${SUPERVISOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      cache: 'no-store'
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return NextResponse.json({ error: 'MT5 candles endpoint failed', detail: text }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach MT5 candles endpoint' }, { status: 502 })
  }
}
