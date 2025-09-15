"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/hooks/use-toast"
import TradingChart from "@/components/TradingChart"
import BackendControlPanel from "@/components/BackendControlPanel"
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign, 
  AlertTriangle,
  Play,
  Pause,
  Square,
  RefreshCw,
  BarChart3,
  Layers,
  Target,
  Settings
} from "lucide-react"

interface SystemStatus {
  system_status: string
  active_positions: number
  daily_pnl: number
  total_volume: number
  uptime: string
  market_session: string
  gemini_status: string
  total_trades: number
  win_rate: number
  last_signal_time: string
}

interface PerformanceAnalytics {
  totalPnL: number
  winRate: number
  sharpeRatio: number
  maxDrawdown: number
  winningTrades: number
  losingTrades: number
  avgWin: number
  avgLoss: number
  startingBalance: number
  currentBalance: number
  dailyChange: number
  monthlyReturn: number
} 

interface Trade {
  id: string
  symbol: string
  direction: string
  volume: number
  entry_price: number
  current_price: number
  profit_loss: number
  status: string
  created_at: string
}

interface Signal {
  id: string
  symbol: string
  direction: string
  entry_price: number
  stop_loss: number
  take_profit: number
  target_price: number
  confidence: number
  strategy: string
  status: string
  created_at: string
  generated_at: string
}

export default function TradingDashboard() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [activeTrades, setActiveTrades] = useState<Trade[]>([])
  const [activeSignals, setActiveSignals] = useState<Signal[]>([])
  const [analytics, setAnalytics] = useState<PerformanceAnalytics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isTrading, setIsTrading] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    fetchSystemData()
    const interval = setInterval(fetchSystemData, 5000) // Update every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchSystemData = async () => {
    try {
      // Fetch system status via Next.js API proxy
      const statusResponse = await fetch('/api/status')
      
      if (statusResponse.ok) {
        const statusData = await statusResponse.json()
        setSystemStatus(statusData)
      } else {
        setSystemStatus(null)
        toast({
          variant: "destructive",
          title: "Failed to load system status",
          description: "Please ensure the backend is running."
        })
      }

      // Fetch active trades via Next.js API proxy
      const tradesResponse = await fetch('/api/trades')
      
      if (tradesResponse.ok) {
        const tradesData = await tradesResponse.json()
        setActiveTrades(tradesData.trades || [])
      } else {
        setActiveTrades([])
        toast({
          variant: "destructive",
          title: "Failed to load active trades",
          description: "The backend did not return active trades."
        })
      }

      // Fetch active signals via Next.js API proxy
      const signalsResponse = await fetch('/api/signals')
      if (signalsResponse.ok) {
        const signalsData = await signalsResponse.json()
        setActiveSignals(signalsData.signals || [])
      } else {
        setActiveSignals([])
      }

      // Fetch performance analytics
      const analyticsResponse = await fetch('/api/analytics/performance')
      if (analyticsResponse.ok) {
        const analyticsData = await analyticsResponse.json()
        setAnalytics(analyticsData)
      }
      
      setIsLoading(false)
    } catch (error) {
      setSystemStatus(null)
      setActiveTrades([])
      setActiveSignals([])
      toast({
        variant: "destructive",
        title: "Error loading data",
        description: "Please ensure the backend is running and reachable."
      })
      setIsLoading(false)
    }
  }

  const handleTradingToggle = async () => {
    try {
      const endpoint = isTrading ? '/api/system/pause' : '/api/system/resume'
      const response = await fetch(endpoint, { method: 'POST' })
      
      if (response.ok) {
        setIsTrading(!isTrading)
        toast({
          title: isTrading ? 'Trading paused' : 'Trading started',
        })
      } else {
        toast({
          variant: 'destructive',
          title: 'Failed to toggle trading',
          description: 'Please try again.'
        })
      }
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Error toggling trading',
        description: 'Please check your connection.'
      })
    }
  }

  const handleKillSwitch = async () => {
    try {
      const response = await fetch('/api/system/kill-switch', { method: 'POST' })
      
      if (response.ok) {
        toast({
          title: 'Kill switch activated',
          description: 'All positions closed and trading stopped.'
        })
        setIsTrading(false)
        // Refresh data to show updated status
        fetchSystemData()
      } else {
        toast({
          variant: 'destructive',
          title: 'Failed to activate kill switch',
          description: 'Please try again.'
        })
      }
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Error activating kill switch',
        description: 'Please check your connection.'
      })
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <RefreshCw className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-green-500/30 pb-4">
          <div>
            <h1 className="text-3xl font-bold text-green-400 font-mono">► ARIA_ELITE.EXE</h1>
            <div className="text-green-300/70 text-sm font-mono mt-1">[INSTITUTIONAL] Neural trading matrix</div>
          </div>
          <div className="flex space-x-2">
            <Button
              onClick={handleTradingToggle}
              variant={systemStatus?.system_status === 'active' ? "destructive" : "default"}
              className="flex items-center space-x-2 bg-red-900/50 hover:bg-red-800/50 border-red-500/50 text-red-300 font-mono"
            >
              {systemStatus?.system_status === 'active' ? (
                <>
                  <Pause className="h-4 w-4" />
                  <span>PAUSE_SYS</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>INIT_SYS</span>
                </>
              )}
            </Button>
            
            <Button
              onClick={handleKillSwitch}
              variant="destructive"
              className="flex items-center space-x-2 bg-red-900 hover:bg-red-800 border-red-500 text-red-100 font-mono animate-pulse"
            >
              <Square className="h-4 w-4" />
              <span>KILL_ALL</span>
            </Button>
            
            <Button
              onClick={fetchSystemData}
              variant="outline"
              size="sm"
              className="bg-green-900/30 hover:bg-green-800/30 border-green-500/50 text-green-400"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* System Status Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-black/90 border-green-500/30 shadow-lg shadow-green-500/20">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-green-400 font-mono">PROFIT_CALC</CardTitle>
              <DollarSign className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold font-mono ${
                (systemStatus?.daily_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"
              }`}>
                ${systemStatus?.daily_pnl?.toFixed(2) || "0.00"}
              </div>
              <p className="text-xs text-green-300/70 font-mono">
                DAILY_DELTA
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/90 border-cyan-500/30 shadow-lg shadow-cyan-500/20">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-cyan-400 font-mono">ACTIVE_CONN</CardTitle>
              <Activity className="h-4 w-4 text-cyan-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-cyan-400 font-mono">
                {systemStatus?.active_positions || 0}
              </div>
              <p className="text-xs text-cyan-300/70 font-mono">
                {systemStatus?.total_trades || 0} TOTAL_EXEC
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/90 border-purple-500/30 shadow-lg shadow-purple-500/20">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-purple-400 font-mono">SUCCESS_RATE</CardTitle>
              <TrendingUp className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-400 font-mono">
                {systemStatus?.win_rate || 0}%
              </div>
              <p className="text-xs text-purple-300/70 font-mono">
                WIN_PROBABILITY
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/90 border-red-500/30 shadow-lg shadow-red-500/20">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-red-400 font-mono">SYS_STATUS</CardTitle>
              <AlertTriangle className="h-4 w-4 text-red-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-400 font-mono">
                {systemStatus?.system_status || "OFFLINE"}
              </div>
              <p className="text-xs text-red-300/70 font-mono">
                LAST_PING: {systemStatus?.last_signal_time ? 
                  new Date(systemStatus.last_signal_time).toLocaleTimeString() : "NEVER"}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Matrix */}
        <Tabs defaultValue="chart" className="space-y-6">
          <TabsList className="bg-black/90 border border-green-500/30">
            <TabsTrigger value="chart" className="data-[state=active]:bg-green-900/50 data-[state=active]:text-green-400 text-green-300/70 font-mono">
              <BarChart3 className="h-4 w-4 mr-2" />
              NEURAL_CHARTS
            </TabsTrigger>
            <TabsTrigger value="trades" className="data-[state=active]:bg-cyan-900/50 data-[state=active]:text-cyan-400 text-cyan-300/70 font-mono">
              <Activity className="h-4 w-4 mr-2" />
              LIVE_TRADES
            </TabsTrigger>
            <TabsTrigger value="signals" className="data-[state=active]:bg-purple-900/50 data-[state=active]:text-purple-400 text-purple-300/70 font-mono">
              <Target className="h-4 w-4 mr-2" />
              AI_SIGNALS
            </TabsTrigger>
            <TabsTrigger value="microstructure" className="data-[state=active]:bg-yellow-900/50 data-[state=active]:text-yellow-400 text-yellow-300/70 font-mono">
              <Layers className="h-4 w-4 mr-2" />
              MICROSTRUCTURE
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-blue-900/50 data-[state=active]:text-blue-400 text-blue-300/70 font-mono">ANALYTICS</TabsTrigger>
            <TabsTrigger value="backend" className="data-[state=active]:bg-red-900/50 data-[state=active]:text-red-400 text-red-300/70 font-mono">
              <Settings className="h-4 w-4 mr-2" />
              BACKEND_CTL
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chart" className="space-y-6">
            <TradingChart />
          </TabsContent>

          <TabsContent value="trades" className="space-y-6">
            <Card className="bg-black/90 border-cyan-500/30 shadow-lg shadow-cyan-500/20">
              <CardHeader>
                <CardTitle className="text-cyan-400 font-mono">► LIVE_TRADES.EXE</CardTitle>
                <CardDescription className="text-cyan-300/70">
                  [REAL-TIME] Active position matrix
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {activeTrades.length > 0 ? activeTrades.map((trade) => (
                    <div key={trade.id} className="flex items-center justify-between p-4 border border-cyan-500/30 rounded-lg bg-black/50">
                      <div className="flex items-center space-x-4">
                        <div className={`p-2 rounded-full ${
                          trade.direction === "buy" ? "bg-green-900/50 border border-green-500/50" : "bg-red-900/50 border border-red-500/50"
                        }`}>
                          {trade.direction === "buy" ? (
                            <TrendingUp className="h-4 w-4 text-green-400" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-red-400" />
                          )}
                        </div>
                        <div>
                          <div className="font-medium text-cyan-400 font-mono">{trade.symbol}</div>
                          <div className="text-sm text-cyan-300/70 font-mono">
                            {trade.direction.toUpperCase()} {trade.volume} LOTS
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-cyan-400 font-mono">
                          ENTRY: {trade.entry_price}
                        </div>
                        <div className="text-sm text-cyan-300/70 font-mono">
                          CURRENT: {trade.current_price}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-medium font-mono ${
                          trade.profit_loss >= 0 ? "text-green-400" : "text-red-400"
                        }`}>
                          ${trade.profit_loss.toFixed(2)}
                        </div>
                        <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 font-mono">{trade.status}</Badge>
                      </div>
                    </div>
                  )) : (
                    <div className="text-center py-8 text-cyan-400/70 font-mono">
                      <div className="animate-pulse mb-2">SCANNING POSITIONS...</div>
                      <div className="text-xs">Live trade data loading from MT5</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="signals" className="space-y-6">
            <Card className="bg-black/90 border-purple-500/30 shadow-lg shadow-purple-500/20">
              <CardHeader>
                <CardTitle className="text-purple-400 font-mono">► AI_SIGNALS.EXE</CardTitle>
                <CardDescription className="text-purple-300/70">
                  [NEURAL] AI-generated trading signals matrix
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {activeSignals.length > 0 ? activeSignals.map((signal) => (
                    <div key={signal.id} className="flex items-center justify-between p-4 border border-purple-500/30 rounded-lg bg-black/50">
                      <div className="flex items-center space-x-4">
                        <div className={`p-2 rounded-full ${
                          signal.direction === "buy" ? "bg-green-900/50 border border-green-500/50" : "bg-red-900/50 border border-red-500/50"
                        }`}>
                          {signal.direction === "buy" ? (
                            <TrendingUp className="h-4 w-4 text-green-400" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-red-400" />
                          )}
                        </div>
                        <div>
                          <div className="font-medium text-purple-400 font-mono">{signal.symbol}</div>
                          <div className="text-sm text-purple-300/70 font-mono">
                            {signal.strategy.toUpperCase()} • {signal.confidence * 100}% CONFIDENCE
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-purple-400 font-mono">
                          ENTRY: {signal.entry_price}
                        </div>
                        <div className="text-sm text-purple-300/70 font-mono">
                          TARGET: {signal.target_price}
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={signal.status === "active" ? "default" : "secondary"} 
                               className="border-purple-500/50 text-purple-400 font-mono">
                          {signal.status}
                        </Badge>
                        <div className="text-xs text-purple-300/70 mt-1 font-mono">
                          {new Date(signal.generated_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  )) : (
                    <div className="text-center py-8 text-purple-400/70 font-mono">
                      <div className="animate-pulse mb-2">ANALYZING NEURAL PATTERNS...</div>
                      <div className="text-xs">AI signals generating from backend models</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="microstructure" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Smart Money Concepts - Real Data Only */}
              <Card className="bg-black/90 border-yellow-500/30 shadow-lg shadow-yellow-500/20">
                <CardHeader>
                  <CardTitle className="text-yellow-400 font-mono">► SMART_MONEY.EXE</CardTitle>
                  <CardDescription className="text-yellow-300/70">
                    [INSTITUTIONAL] Smart money pattern detection
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center py-8 text-yellow-400/70 font-mono">
                    <div className="animate-pulse mb-2">SCANNING INSTITUTIONAL FLOW...</div>
                    <div className="text-xs">Smart money data loading from market analysis</div>
                  </div>
                </CardContent>
              </Card>


              {/* Order Flow Analysis - Real Data Only */}
              <Card className="bg-black/90 border-green-500/30 shadow-lg shadow-green-500/20">
                <CardHeader>
                  <CardTitle className="text-green-400 font-mono">► ORDER_FLOW.EXE</CardTitle>
                  <CardDescription className="text-green-300/70">
                    [REAL-TIME] Institutional activity matrix
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center py-8 text-green-400/70 font-mono">
                    <div className="animate-pulse mb-2">SCANNING NETWORKS...</div>
                    <div className="text-xs">Live order flow data will load from MT5 feed</div>
                  </div>
                </CardContent>
              </Card>

              {/* Market Regime - Real Data Only */}
              <Card className="bg-black/90 border-cyan-500/30 shadow-lg shadow-cyan-500/20">
                <CardHeader>
                  <CardTitle className="text-cyan-400 font-mono">► REGIME_SCAN.EXE</CardTitle>
                  <CardDescription className="text-cyan-300/70">
                    [NEURAL] Market state classification engine
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center py-8 text-cyan-400/70 font-mono">
                    <div className="animate-pulse mb-2">ANALYZING PATTERNS...</div>
                    <div className="text-xs">Market regime data loading from AI models</div>
                  </div>
                </CardContent>
              </Card>

              {/* Liquidity Analysis - Real Data Only */}
              <Card className="bg-black/90 border-purple-500/30 shadow-lg shadow-purple-500/20">
                <CardHeader>
                  <CardTitle className="text-purple-400 font-mono">► LIQUIDITY_PROBE.EXE</CardTitle>
                  <CardDescription className="text-purple-300/70">
                    [DEPTH] Market liquidity analysis matrix
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center py-8 text-purple-400/70 font-mono">
                    <div className="animate-pulse mb-2">PROBING DEPTH...</div>
                    <div className="text-xs">Live liquidity data loading from MT5 market depth</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <Card className="bg-black/90 border-green-500/30 shadow-lg shadow-green-500/20">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-green-400 font-mono">
                  <BarChart3 className="h-5 w-5" />
                  <span>► PERFORMANCE_ANALYTICS.EXE</span>
                </CardTitle>
                <CardDescription className="text-green-300/70 font-mono">
                  [REAL-TIME] Trading performance metrics and analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card className="bg-black/50 border-green-500/30">
                    <CardContent className="p-4">
                      <div className="text-green-300/70 font-mono text-xs">TOTAL_P&L</div>
                      <div className="text-2xl font-bold text-green-400 font-mono">
                        ${analytics?.totalPnL?.toLocaleString()}
                      </div>
                      <div className="text-green-300/70 font-mono text-xs">MT5_FEED</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-black/50 border-cyan-500/30">
                    <CardContent className="p-4">
                      <div className="text-cyan-300/70 font-mono text-xs">WIN_RATE</div>
                      <div className="text-2xl font-bold text-cyan-400 font-mono">
                        {analytics?.winRate}%
                      </div>
                      <div className="text-cyan-300/70 font-mono text-xs">ALGORITHMIC</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-black/50 border-purple-500/30">
                    <CardContent className="p-4">
                      <div className="text-purple-300/70 font-mono text-xs">SHARPE_RATIO</div>
                      <div className="text-2xl font-bold text-purple-400 font-mono">
                        {analytics?.sharpeRatio?.toFixed(2)}
                      </div>
                      <div className="text-purple-300/70 font-mono text-xs">RISK_ADJUSTED</div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-black/50 border-yellow-500/30">
                    <CardContent className="p-4">
                      <div className="text-yellow-300/70 font-mono text-xs">MAX_DRAWDOWN</div>
                      <div className="text-2xl font-bold text-yellow-400 font-mono">
                        {analytics?.maxDrawdown}%
                      </div>
                      <div className="text-yellow-300/70 font-mono text-xs">RISK_METRIC</div>
                    </CardContent>
                  </Card>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-black/50 border-green-500/30">
                    <CardHeader>
                      <CardTitle className="text-green-400 font-mono text-sm">► EQUITY_CURVE</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-green-300/70 font-mono text-xs">STARTING_BALANCE:</span>
                          <span className="text-green-400 font-mono">${analytics?.startingBalance?.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-green-300/70 font-mono text-xs">CURRENT_BALANCE:</span>
                          <span className="text-green-400 font-mono">${analytics?.currentBalance?.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-green-300/70 font-mono text-xs">DAILY_CHANGE:</span>
                          <span className={`font-mono ${(analytics?.dailyChange ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {(analytics?.dailyChange ?? 0) >= 0 ? '+' : ''}${analytics?.dailyChange?.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-green-300/70 font-mono text-xs">MONTHLY_RETURN:</span>
                          <span className={`font-mono ${(analytics?.monthlyReturn ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {analytics?.monthlyReturn?.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-black/50 border-cyan-500/30">
                    <CardHeader>
                      <CardTitle className="text-cyan-400 font-mono text-sm">► TRADE_DISTRIBUTION</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-cyan-300/70 font-mono text-xs">WINNING_TRADES:</span>
                          <span className="text-cyan-400 font-mono">{analytics?.winningTrades}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-cyan-300/70 font-mono text-xs">LOSING_TRADES:</span>
                          <span className="text-cyan-400 font-mono">{analytics?.losingTrades}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-cyan-300/70 font-mono text-xs">AVG_WIN:</span>
                          <span className="text-cyan-400 font-mono">${analytics?.avgWin?.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-cyan-300/70 font-mono text-xs">AVG_LOSS:</span>
                          <span className="text-cyan-400 font-mono">${analytics?.avgLoss?.toFixed(2)}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="backend" className="space-y-6">
            <BackendControlPanel />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}