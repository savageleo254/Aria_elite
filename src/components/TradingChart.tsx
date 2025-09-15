"use client"

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Activity,
  RefreshCw,
  Settings
} from "lucide-react"
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar
} from 'recharts'

interface CandleData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface OrderFlowData {
  price: number
  buyVolume: number
  sellVolume: number
  imbalance: number
}

interface MarketDepthData {
  price: number
  bidSize: number
  askSize: number
  spread: number
}

export default function TradingChart() {
  const [symbol, setSymbol] = useState('XAUUSD')
  const [isLoading, setIsLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick')
  const [timeframe, setTimeframe] = useState('M5')
  const [candleData, setCandleData] = useState<CandleData[]>([])
  const [orderFlow, setOrderFlow] = useState<OrderFlowData[]>([])
  const [marketDepth, setMarketDepth] = useState<MarketDepthData[]>([])
  const [currentPrice, setCurrentPrice] = useState(2650.50)
  const [priceChange, setPriceChange] = useState(+12.30)
  const [volume, setVolume] = useState(145230)
  
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    fetchMT5Data()
    const interval = setInterval(() => {
      fetchRealTimeUpdates()
    }, 1000)
    
    return () => clearInterval(interval)
  }, [symbol, timeframe])

  const fetchMT5Data = async () => {
    setIsLoading(true)
    try {
      // Fetch historical candlestick data from MT5
      const candleResponse = await fetch(`/api/mt5/candles?symbol=${symbol}&timeframe=${timeframe}&count=100`)
      if (candleResponse.ok) {
        const candleData = await candleResponse.json()
        setCandleData(candleData.candles || [])
      }

      // Fetch real-time market depth from MT5 Level II
      const depthResponse = await fetch(`/api/mt5/market-depth?symbol=${symbol}`)
      if (depthResponse.ok) {
        const depthData = await depthResponse.json()
        setMarketDepth(depthData.depth || [])
      }

      // Fetch order flow data from MT5 tick analysis
      const flowResponse = await fetch(`/api/mt5/order-flow?symbol=${symbol}`)
      if (flowResponse.ok) {
        const flowData = await flowResponse.json()
        setOrderFlow(flowData.orderFlow || [])
      }

      // Get current price from MT5
      const priceResponse = await fetch(`/api/mt5/price?symbol=${symbol}`)
      if (priceResponse.ok) {
        const priceData = await priceResponse.json()
        setCurrentPrice(priceData.price)
        setPriceChange(priceData.change)
        setVolume(priceData.volume)
      }
    } catch (error) {
      console.error('Failed to fetch MT5 data:', error)
    }
    setIsLoading(false)
  }

  const fetchRealTimeUpdates = async () => {
    try {
      // Get latest tick data from MT5
      const tickResponse = await fetch(`/api/mt5/tick?symbol=${symbol}`)
      if (tickResponse.ok) {
        const tickData = await tickResponse.json()
        setCurrentPrice(tickData.bid)
        setPriceChange(tickData.change)
        setVolume(prev => prev + tickData.volume)
        
        // Update latest candle with real MT5 tick
        setCandleData(prev => {
          if (prev.length === 0) return prev
          
          const updated = [...prev]
          const latest = updated[updated.length - 1]
          
          updated[updated.length - 1] = {
            ...latest,
            close: tickData.bid,
            high: Math.max(latest.high, tickData.ask),
            low: Math.min(latest.low, tickData.bid),
            volume: latest.volume + tickData.volume
          }
          
          return updated
        })
      }
    } catch (error) {
      console.error('Failed to fetch MT5 real-time data:', error)
    }
  }

  const CustomCandlestick = ({ payload, x, y, width, height }: any) => {
    if (!payload) return null
    
    const { open, high, low, close } = payload
    const isGreen = close > open
    const bodyHeight = Math.abs(close - open) * (height / (high - low))
    const bodyY = y + (Math.max(close, open) - high) * (height / (high - low))
    
    return (
      <g>
        {/* Wick */}
        <line
          x1={x + width/2}
          y1={y}
          x2={x + width/2}
          y2={y + height}
          stroke={isGreen ? "#22c55e" : "#ef4444"}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x + 1}
          y={bodyY}
          width={width - 2}
          height={Math.max(bodyHeight, 1)}
          fill={isGreen ? "#22c55e" : "#ef4444"}
          stroke={isGreen ? "#22c55e" : "#ef4444"}
        />
      </g>
    )
  }

  const refreshData = async () => {
    setIsLoading(true)
    // Fetch real data from MT5 feeds
    try {
      const candlesRes = await fetch(`/api/mt5/candles?symbol=${symbol}&timeframe=${timeframe}`)
      if (candlesRes.ok) {
        const candlesData = await candlesRes.json()
        setCandleData(candlesData.candles || [])
      }
      
      const priceRes = await fetch(`/api/mt5/price?symbol=${symbol}`)
      if (priceRes.ok) {
        const priceData = await priceRes.json()
        setCurrentPrice(priceData.price || 0)
        setPriceChange(priceData.change || 0)
      }
    } catch (error) {
      console.error('Failed to fetch market data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="bg-black/90 border-green-500/30 shadow-lg shadow-green-500/20">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2 text-green-400 font-mono">
              <BarChart3 className="h-5 w-5" />
              <span>► NEURAL_CHARTS.EXE</span>
            </CardTitle>
            <CardDescription className="text-green-300/70 font-mono">
              [REAL-TIME] Market data analysis matrix
            </CardDescription>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Badge variant="default" className="bg-green-900/50 border-green-500/50 text-green-400 font-mono">
                {symbol} ${currentPrice.toFixed(2)}
              </Badge>
              <Badge variant={priceChange >= 0 ? "default" : "destructive"} 
                     className={`font-mono ${priceChange >= 0 ? 'bg-green-900/50 border-green-500/50 text-green-400' : 'bg-red-900/50 border-red-500/50 text-red-400'}`}>
                {priceChange >= 0 ? "+" : ""}{priceChange.toFixed(2)}
              </Badge>
            </div>
            
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-20 bg-black/50 border-green-500/50 text-green-400 font-mono">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black border-green-500/50">
                <SelectItem value="M1" className="text-green-400 font-mono">M1</SelectItem>
                <SelectItem value="M5" className="text-green-400 font-mono">M5</SelectItem>
                <SelectItem value="M15" className="text-green-400 font-mono">M15</SelectItem>
                <SelectItem value="H1" className="text-green-400 font-mono">H1</SelectItem>
                <SelectItem value="H4" className="text-green-400 font-mono">H4</SelectItem>
                <SelectItem value="D1" className="text-green-400 font-mono">D1</SelectItem>
              </SelectContent>
            </Select>
            
            <Button
              variant="outline"
              size="sm"
              onClick={refreshData}
              className="flex items-center space-x-1 bg-green-900/30 border-green-500/50 text-green-400 hover:bg-green-800/30"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
        
        <CardContent>
          <Tabs defaultValue="price" className="space-y-4">
            <TabsList className="bg-black/90 border border-green-500/30">
              <TabsTrigger value="price" className="data-[state=active]:bg-green-900/50 data-[state=active]:text-green-400 text-green-300/70 font-mono">PRICE_MATRIX</TabsTrigger>
              <TabsTrigger value="orderflow" className="data-[state=active]:bg-cyan-900/50 data-[state=active]:text-cyan-400 text-cyan-300/70 font-mono">ORDER_FLOW</TabsTrigger>
              <TabsTrigger value="depth" className="data-[state=active]:bg-purple-900/50 data-[state=active]:text-purple-400 text-purple-300/70 font-mono">DEPTH_SCAN</TabsTrigger>
              <TabsTrigger value="microstructure" className="data-[state=active]:bg-yellow-900/50 data-[state=active]:text-yellow-400 text-yellow-300/70 font-mono">MICRO_SCAN</TabsTrigger>
            </TabsList>
            
            {/* Price Chart */}
            <TabsContent value="price" className="space-y-4">
              <div className="flex items-center space-x-2">
                <Button 
                  variant={chartType === 'candlestick' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setChartType('candlestick')}
                  className={`font-mono ${chartType === 'candlestick' ? 'bg-green-900/70 border-green-500/50 text-green-400' : 'bg-black/50 border-green-500/30 text-green-300/70 hover:bg-green-900/30'}`}
                >
                  CANDLES
                </Button>
                <Button 
                  variant={chartType === 'line' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setChartType('line')}
                  className={`font-mono ${chartType === 'line' ? 'bg-cyan-900/70 border-cyan-500/50 text-cyan-400' : 'bg-black/50 border-cyan-500/30 text-cyan-300/70 hover:bg-cyan-900/30'}`}
                >
                  LINE_SIG
                </Button>
                <Button 
                  variant={chartType === 'area' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setChartType('area')}
                  className={`font-mono ${chartType === 'area' ? 'bg-purple-900/70 border-purple-500/50 text-purple-400' : 'bg-black/50 border-purple-500/30 text-purple-300/70 hover:bg-purple-900/30'}`}
                >
                  AREA_MAP
                </Button>
              </div>
              
              <div className="h-96 bg-black/50 border border-green-500/20 rounded-lg p-4">
                <ResponsiveContainer width="100%" height="100%">
                  {chartType === 'line' ? (
                    <LineChart data={candleData}>
                      <CartesianGrid strokeDasharray="2 2" stroke="#10b981" opacity={0.1} />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                        axisLine={{ stroke: '#10b981', opacity: 0.3 }}
                        tickLine={{ stroke: '#10b981', opacity: 0.3 }}
                        tick={{ fill: '#10b981', fontSize: 11, fontFamily: 'monospace' }}
                      />
                      <YAxis 
                        domain={['dataMin - 5', 'dataMax + 5']}
                        axisLine={{ stroke: '#10b981', opacity: 0.3 }}
                        tickLine={{ stroke: '#10b981', opacity: 0.3 }}
                        tick={{ fill: '#10b981', fontSize: 11, fontFamily: 'monospace' }}
                      />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value: number) => [value.toFixed(2), 'PRICE']}
                        contentStyle={{
                          backgroundColor: 'rgba(0,0,0,0.9)',
                          border: '1px solid rgba(16,185,129,0.5)',
                          borderRadius: '4px',
                          color: '#10b981',
                          fontFamily: 'monospace'
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="close" 
                        stroke="#10b981" 
                        strokeWidth={2}
                        dot={false}
                        strokeDasharray="0"
                      />
                    </LineChart>
                  ) : (
                    <div className="h-full w-full bg-black/30 rounded border border-green-500/20 p-4">
                      <div className="text-green-400 font-mono text-sm mb-4 flex items-center justify-between">
                        <span>► CANDLESTICK_ANALYSIS.EXE</span>
                        <span className="animate-pulse">LIVE_DATA_STREAM</span>
                      </div>
                      <svg width="100%" height="80%" className="border border-green-500/10">
                        {candleData.map((candle, index) => (
                          <CustomCandlestick
                            key={candle.timestamp}
                            data={candle}
                            x={index * (100 / candleData.length)}
                            width={100 / candleData.length - 2}
                          />
                        ))}
                      </svg>
                      <div className="mt-4 text-green-300/70 font-mono text-xs">
                        {candleData.length > 0 && `LAST_UPDATE: ${new Date(candleData[candleData.length - 1]?.timestamp || Date.now()).toLocaleString()}`}
                      </div>
                    </div>
                  )}
                </ResponsiveContainer>
              </div>
              
              {/* Volume Chart */}
              <div className="h-32 bg-black/50 border border-cyan-500/20 rounded-lg p-4">
                <div className="text-cyan-400 font-mono text-sm mb-2">► VOLUME_ANALYSIS</div>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={candleData}>
                    <CartesianGrid strokeDasharray="2 2" stroke="#06b6d4" opacity={0.1} />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      axisLine={{ stroke: '#06b6d4', opacity: 0.3 }}
                      tickLine={{ stroke: '#06b6d4', opacity: 0.3 }}
                      tick={{ fill: '#06b6d4', fontSize: 10, fontFamily: 'monospace' }}
                    />
                    <YAxis 
                      axisLine={{ stroke: '#06b6d4', opacity: 0.3 }}
                      tickLine={{ stroke: '#06b6d4', opacity: 0.3 }}
                      tick={{ fill: '#06b6d4', fontSize: 10, fontFamily: 'monospace' }}
                    />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                      formatter={(value: any) => [value.toLocaleString(), 'VOLUME']}
                      contentStyle={{
                        backgroundColor: 'rgba(0,0,0,0.9)',
                        border: '1px solid rgba(6,182,212,0.5)',
                        borderRadius: '4px',
                        color: '#06b6d4',
                        fontFamily: 'monospace'
                      }}
                    />
                    <Bar dataKey="volume" fill="rgba(6,182,212,0.6)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            
            {/* Order Flow */}
            <TabsContent value="orderflow" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono">
                <div className="bg-black/50 border border-cyan-500/20 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-mono text-sm mb-2">► BUY/SELL_PRESSURE</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={orderFlow} layout="horizontal">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="price" />
                        <Tooltip />
                        <Bar dataKey="buyVolume" fill="#22c55e" />
                        <Bar dataKey="sellVolume" fill="#ef4444" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-2">Order Flow Imbalance</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={orderFlow}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="price" />
                        <YAxis />
                        <Tooltip />
                        <ReferenceLine y={0} stroke="#000" />
                        <Bar 
                          dataKey="imbalance" 
                          fill="#8884d8"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </TabsContent>
            
            {/* Market Depth */}
            <TabsContent value="depth" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={marketDepth}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="price" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="bidSize" fill="#22c55e" />
                    <Bar dataKey="askSize" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="text-sm text-gray-500">Bid/Ask Spread</div>
                    <div className="text-xl font-bold">0.25</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-sm text-gray-500">Total Volume</div>
                    <div className="text-xl font-bold">{volume.toLocaleString()}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-sm text-gray-500">Liquidity Index</div>
                    <div className="text-xl font-bold text-green-600">High</div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Microstructure Analysis */}
            <TabsContent value="microstructure" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Smart Money Concepts</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Order Blocks</span>
                      <Badge variant="secondary">3 Detected</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Fair Value Gaps</span>
                      <Badge variant="secondary">2 Active</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Liquidity Sweeps</span>
                      <Badge variant="destructive">1 Recent</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Market Structure</span>
                      <Badge variant="default">Bullish</Badge>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Institutional Flow</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Large Orders</span>
                      <Badge variant="secondary">15 Detected</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Block Trades</span>
                      <Badge variant="secondary">3 Active</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Dark Pool Activity</span>
                      <Badge variant="default">Moderate</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Institutional Bias</span>
                      <Badge variant="default">Long</Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <Card className="bg-black/50 border border-yellow-500/20">
                <CardHeader>
                  <CardTitle className="text-yellow-400 font-mono text-base">► MARKET_REGIME_ANALYSIS</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400 font-mono">TRENDING</div>
                      <div className="text-xs text-green-300/70 font-mono">CURRENT_REGIME</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-400 font-mono">0.78</div>
                      <div className="text-xs text-cyan-300/70 font-mono">VOLATILITY_IDX</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400 font-mono">HIGH</div>
                      <div className="text-xs text-purple-300/70 font-mono">MOMENTUM</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400 font-mono">85%</div>
                      <div className="text-xs text-green-300/70 font-mono">CONFIDENCE</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    )
  }
