"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  BarChart3,
  Zap,
  Target,
  Clock,
  DollarSign,
  RefreshCw,
  AlertTriangle
} from "lucide-react"

interface OrderFlowData {
  symbol: string
  timestamp: string
  buy_volume: number
  sell_volume: number
  imbalance_ratio: number
  vwap: string
  twap: string
  participation_rate: number
  market_impact: number
}

interface LiquidityData {
  symbol: string
  timestamp: string
  bid_ask_spread: string
  spread_bps: number
  market_depth: {
    bid_depth_level1: number
    ask_depth_level1: number
    bid_depth_level5: number
    ask_depth_level5: number
  }
  liquidity_score: number
  slippage_estimate: number
}

interface ExecutionQuality {
  symbol: string
  timestamp: string
  average_slippage: number
  fill_rate: number
  reject_rate: number
  execution_latency_ms: number
  price_improvement_rate: number
  effective_spread: number
  implementation_shortfall: number
}

interface TickData {
  timestamp: string
  bid: number
  ask: number
  last: number
  volume: number
  side: string
}

interface MarketImpact {
  symbol: string
  order_size: number
  side: string
  estimated_impact_pips: number
  estimated_impact_percentage: number
  confidence: number
  timestamp: string
}

export default function MarketMicrostructureDashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState("EURUSD")
  const [orderFlowData, setOrderFlowData] = useState<OrderFlowData | null>(null)
  const [liquidityData, setLiquidityData] = useState<LiquidityData | null>(null)
  const [executionQuality, setExecutionQuality] = useState<ExecutionQuality | null>(null)
  const [recentTicks, setRecentTicks] = useState<TickData[]>([])
  const [marketImpact, setMarketImpact] = useState<MarketImpact | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

  useEffect(() => {
    fetchMicrostructureData()
    
    let interval: NodeJS.Timeout
    if (autoRefresh) {
      interval = setInterval(fetchMicrostructureData, 2000) // Update every 2 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [selectedSymbol, autoRefresh])

  const fetchMicrostructureData = async () => {
    try {
      const [orderFlowRes, liquidityRes, execQualityRes, ticksRes] = await Promise.all([
        fetch(`/api/microstructure/order-flow/${selectedSymbol}`),
        fetch(`/api/microstructure/liquidity/${selectedSymbol}`),
        fetch(`/api/microstructure/execution-quality/${selectedSymbol}`),
        fetch(`/api/microstructure/live-ticks/${selectedSymbol}?limit=50`)
      ])

      if (orderFlowRes.ok) {
        setOrderFlowData(await orderFlowRes.json())
      }
      if (liquidityRes.ok) {
        setLiquidityData(await liquidityRes.json())
      }
      if (execQualityRes.ok) {
        setExecutionQuality(await execQualityRes.json())
      }
      if (ticksRes.ok) {
        const tickData = await ticksRes.json()
        setRecentTicks(tickData.ticks || [])
      }

      // Calculate market impact for standard lot
      const impactRes = await fetch(`/api/microstructure/market-impact/${selectedSymbol}?order_size=100000&side=buy`)
      if (impactRes.ok) {
        setMarketImpact(await impactRes.json())
      }

      setIsLoading(false)
    } catch (error) {
      console.error('Error fetching microstructure data:', error)
      setIsLoading(false)
    }
  }

  const formatNumber = (num: number, decimals: number = 2) => {
    return num?.toFixed(decimals) || "0.00"
  }

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`
    return volume.toString()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <RefreshCw className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-8xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Market Microstructure Analysis</h1>
            <p className="text-gray-600">Real-time order flow, liquidity, and execution analytics</p>
          </div>
          <div className="flex items-center space-x-4">
            <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {symbols.map(symbol => (
                  <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={() => setAutoRefresh(!autoRefresh)}
              variant={autoRefresh ? "default" : "outline"}
              size="sm"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
              Auto Refresh
            </Button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Order Flow Imbalance</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${
                (orderFlowData?.imbalance_ratio || 0) > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {formatNumber((orderFlowData?.imbalance_ratio || 0) * 100, 1)}%
              </div>
              <p className="text-xs text-muted-foreground">
                Buy: {formatVolume(orderFlowData?.buy_volume || 0)} | 
                Sell: {formatVolume(orderFlowData?.sell_volume || 0)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Bid-Ask Spread</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatNumber(liquidityData?.spread_bps || 0, 1)} bps
              </div>
              <p className="text-xs text-muted-foreground">
                Raw spread: {liquidityData?.bid_ask_spread || "0.00000"}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Liquidity Score</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatNumber((liquidityData?.liquidity_score || 0) * 100, 0)}%
              </div>
              <Progress 
                value={(liquidityData?.liquidity_score || 0) * 100} 
                className="mt-2"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Execution Latency</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatNumber(executionQuality?.execution_latency_ms || 0, 1)}ms
              </div>
              <p className="text-xs text-muted-foreground">
                Fill rate: {formatNumber((executionQuality?.fill_rate || 0) * 100, 1)}%
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Analysis Tabs */}
        <Tabs defaultValue="orderflow" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="orderflow">Order Flow</TabsTrigger>
            <TabsTrigger value="liquidity">Liquidity</TabsTrigger>
            <TabsTrigger value="execution">Execution</TabsTrigger>
            <TabsTrigger value="ticks">Live Ticks</TabsTrigger>
            <TabsTrigger value="impact">Market Impact</TabsTrigger>
          </TabsList>

          <TabsContent value="orderflow" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Volume Analysis</CardTitle>
                  <CardDescription>Order flow volume breakdown</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Buy Volume</span>
                    <span className="text-green-600 font-bold">
                      {formatVolume(orderFlowData?.buy_volume || 0)}
                    </span>
                  </div>
                  <Progress 
                    value={((orderFlowData?.buy_volume || 0) / ((orderFlowData?.buy_volume || 0) + (orderFlowData?.sell_volume || 1))) * 100} 
                    className="bg-red-100"
                  />
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Sell Volume</span>
                    <span className="text-red-600 font-bold">
                      {formatVolume(orderFlowData?.sell_volume || 0)}
                    </span>
                  </div>
                  <div className="pt-4 border-t">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Net Imbalance</span>
                      <Badge variant={
                        (orderFlowData?.imbalance_ratio || 0) > 0.1 ? "default" : 
                        (orderFlowData?.imbalance_ratio || 0) < -0.1 ? "destructive" : "secondary"
                      }>
                        {formatNumber((orderFlowData?.imbalance_ratio || 0) * 100, 1)}%
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Price Benchmarks</CardTitle>
                  <CardDescription>VWAP and TWAP analysis</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">VWAP</span>
                      <span className="font-mono text-sm">
                        {orderFlowData?.vwap || "0.00000"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">TWAP</span>
                      <span className="font-mono text-sm">
                        {orderFlowData?.twap || "0.00000"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Participation Rate</span>
                      <span className="text-sm">
                        {formatNumber((orderFlowData?.participation_rate || 0) * 100, 1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Market Impact</span>
                      <Badge variant={
                        (orderFlowData?.market_impact || 0) > 0.01 ? "destructive" : "default"
                      }>
                        {formatNumber((orderFlowData?.market_impact || 0) * 100, 3)}%
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="liquidity" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Spread Analysis</CardTitle>
                  <CardDescription>Bid-ask spread metrics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Raw Spread</span>
                    <span className="font-mono text-sm">
                      {liquidityData?.bid_ask_spread || "0.00000"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Spread (bps)</span>
                    <span className="text-sm font-bold">
                      {formatNumber(liquidityData?.spread_bps || 0, 1)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Slippage Estimate</span>
                    <Badge variant={
                      (liquidityData?.slippage_estimate || 0) > 0.005 ? "destructive" : "default"
                    }>
                      {formatNumber((liquidityData?.slippage_estimate || 0) * 10000, 1)} pips
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Market Depth</CardTitle>
                  <CardDescription>Liquidity at different levels</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>Level 1 Depth</span>
                        <span>Bid/Ask</span>
                      </div>
                      <div className="flex justify-between text-sm font-medium">
                        <span className="text-green-600">
                          {formatVolume(liquidityData?.market_depth?.bid_depth_level1 || 0)}
                        </span>
                        <span className="text-red-600">
                          {formatVolume(liquidityData?.market_depth?.ask_depth_level1 || 0)}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>Level 5 Depth</span>
                        <span>Bid/Ask</span>
                      </div>
                      <div className="flex justify-between text-sm font-medium">
                        <span className="text-green-600">
                          {formatVolume(liquidityData?.market_depth?.bid_depth_level5 || 0)}
                        </span>
                        <span className="text-red-600">
                          {formatVolume(liquidityData?.market_depth?.ask_depth_level5 || 0)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="pt-4 border-t">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Overall Liquidity</span>
                      <Progress 
                        value={(liquidityData?.liquidity_score || 0) * 100}
                        className="w-24"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="execution" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Execution Metrics</CardTitle>
                  <CardDescription>Order execution performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Average Slippage</span>
                    <Badge variant={
                      (executionQuality?.average_slippage || 0) > 1 ? "destructive" : "default"
                    }>
                      {formatNumber(executionQuality?.average_slippage || 0, 1)} pips
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Fill Rate</span>
                    <span className="text-green-600 font-bold">
                      {formatNumber((executionQuality?.fill_rate || 0) * 100, 1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Reject Rate</span>
                    <span className={`font-bold ${
                      (executionQuality?.reject_rate || 0) > 0.05 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {formatNumber((executionQuality?.reject_rate || 0) * 100, 2)}%
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Latency Metrics</CardTitle>
                  <CardDescription>Order processing speed</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Execution Latency</span>
                    <Badge variant={
                      (executionQuality?.execution_latency_ms || 0) > 50 ? "destructive" : "default"
                    }>
                      {formatNumber(executionQuality?.execution_latency_ms || 0, 1)}ms
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Price Improvement</span>
                    <span className="text-green-600 font-bold">
                      {formatNumber((executionQuality?.price_improvement_rate || 0) * 100, 1)}%
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Cost Analysis</CardTitle>
                  <CardDescription>Trading cost breakdown</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Effective Spread</span>
                    <span className="font-bold">
                      {formatNumber(executionQuality?.effective_spread || 0, 1)} bps
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Implementation Shortfall</span>
                    <Badge variant={
                      (executionQuality?.implementation_shortfall || 0) > 0.01 ? "destructive" : "default"
                    }>
                      {formatNumber((executionQuality?.implementation_shortfall || 0) * 100, 2)}%
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="ticks" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Live Tick Stream ({selectedSymbol})</CardTitle>
                <CardDescription>Real-time tick-by-tick data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {recentTicks.slice(0, 20).map((tick, index) => (
                    <div key={index} className="flex items-center justify-between p-2 border rounded text-sm">
                      <div className="flex items-center space-x-4">
                        <Badge variant={tick.side === 'buy' ? "default" : "destructive"}>
                          {tick.side.toUpperCase()}
                        </Badge>
                        <span className="font-mono">{formatNumber(tick.last, 5)}</span>
                        <span className="text-muted-foreground">
                          Vol: {formatVolume(tick.volume)}
                        </span>
                      </div>
                      <div className="text-right text-xs text-muted-foreground">
                        <div>B: {formatNumber(tick.bid, 5)}</div>
                        <div>A: {formatNumber(tick.ask, 5)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="impact" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Market Impact Calculator</CardTitle>
                <CardDescription>Estimate impact for different order sizes</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {marketImpact && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="font-semibold">Standard Lot (100K) Impact</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Order Size:</span>
                          <span className="font-mono">{formatVolume(marketImpact.order_size)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Side:</span>
                          <Badge variant={marketImpact.side === 'buy' ? "default" : "destructive"}>
                            {marketImpact.side.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Impact (pips):</span>
                          <span className="font-bold text-red-600">
                            {formatNumber(marketImpact.estimated_impact_pips, 2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Impact (%):</span>
                          <span className="font-bold text-red-600">
                            {formatNumber(marketImpact.estimated_impact_percentage, 4)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Confidence:</span>
                          <Progress value={marketImpact.confidence * 100} className="w-24" />
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <h3 className="font-semibold">Impact Guidelines</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                          <span>{"< 0.5 pips: Low Impact"}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                          <span>0.5-2 pips: Medium Impact</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                          <span>{"> 2 pips: High Impact"}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
