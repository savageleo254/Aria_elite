"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { 
  Settings, 
  Brain, 
  Activity, 
  Zap,
  RefreshCw,
  Download,
  Save,
  AlertTriangle,
  Loader,
  BarChart3,
  Cpu,
  Network,
  Bot,
  Plus
} from "lucide-react"

interface ModelStatus {
  name: string
  status: string
  accuracy: number
  last_trained: string
  version: string
  size: string
}

interface WorkflowAgent {
  name: string
  status: string
  last_execution: string
  success_rate: number
  configuration: Record<string, any>
}

interface SystemConfig {
  execution_engine: Record<string, any>
  risk_management: Record<string, any>
  mt5_connection: Record<string, any>
  ai_models: Record<string, any>
}

export default function BackendControlPanel() {
  const [models, setModels] = useState<ModelStatus[]>([])
  const [workflows, setWorkflows] = useState<WorkflowAgent[]>([])
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("models")
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    fetchSystemData()
    const interval = setInterval(fetchSystemData, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchSystemData = async () => {
    try {
      // Fetch model status
      const modelsRes = await fetch('/api/models/status')
      if (modelsRes.ok) {
        const modelsData = await modelsRes.json()
        setModels(modelsData.model_status || [])
      }

      // Fetch workflow agents status
      const workflowsRes = await fetch('/api/agents/status')
      if (workflowsRes.ok) {
        const workflowsData = await workflowsRes.json()
        setWorkflows(workflowsData.agents || [])
      }

      // Fetch system configuration
      const configRes = await fetch('/api/config/system')
      if (configRes.ok) {
        const configData = await configRes.json()
        setSystemConfig(configData.config || {})
      }

      setIsLoading(false)
    } catch (error) {
      console.error('Failed to fetch system data:', error)
      toast({
        variant: "destructive",
        title: "Failed to load system data",
        description: "Please ensure the backend is running."
      })
    }
  }

  const handleModelRetrain = async (modelName?: string) => {
    setIsTraining(true)
    setTrainingProgress(0)
    
    try {
      const endpoint = modelName ? `/api/models/retrain/${modelName}` : '/api/models/retrain'
      const res = await fetch(endpoint, { method: 'POST' })
      
      if (res.ok) {
        toast({
          title: "Training Started",
          description: `Model${modelName ? ` ${modelName}` : 's'} retraining initiated in background`
        })
        
        // Track training progress via backend polling
        const progressInterval = setInterval(async () => {
          try {
            const statusResponse = await fetch('/api/models/training-status')
            if (statusResponse.ok) {
              const status = await statusResponse.json()
              setTrainingProgress(status.progress || 0)
              if (status.completed) {
                clearInterval(progressInterval)
                setIsTraining(false)
                setTrainingProgress(100)
              }
            }
          } catch (error) {
            console.error('Training status check failed:', error)
          }
        }, 2000)
      } else {
        throw new Error('Training request failed')
      }
    } catch (error) {
      setIsTraining(false)
      toast({
        variant: "destructive",
        title: "Training Failed",
        description: "Failed to start model training"
      })
    }
  }

  const handleConfigUpdate = async (section: string, config: Record<string, any>) => {
    try {
      const res = await fetch(`/api/config/${section}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (res.ok) {
        toast({
          title: "Configuration Updated",
          description: `${section} configuration saved successfully`
        })
        fetchSystemData()
      } else {
        throw new Error('Config update failed')
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Failed to update configuration"
      })
    }
  }

  const handleWorkflowToggle = async (agentName: string, enable: boolean) => {
    try {
      const res = await fetch(`/api/agents/${agentName}/${enable ? 'enable' : 'disable'}`, {
        method: 'POST'
      })
      
      if (res.ok) {
        toast({
          title: `Agent ${enable ? 'Enabled' : 'Disabled'}`,
          description: `${agentName} workflow agent status updated`
        })
        fetchSystemData()
      } else {
        throw new Error('Workflow toggle failed')
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Toggle Failed",
        description: "Failed to update workflow agent status"
      })
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[600px]">
        <Loader className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="bg-black/90 border-green-500/30 shadow-lg shadow-green-500/20">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-green-400 font-mono">
            <Settings className="h-5 w-5" />
            <span>► BACKEND_CONTROL_MATRIX.EXE</span>
          </CardTitle>
          <CardDescription className="text-green-300/70 font-mono">
            [INSTITUTIONAL] System management and configuration interface
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-6 bg-black/90 border border-green-500/30">
              <TabsTrigger value="models" className="data-[state=active]:bg-green-900/50 data-[state=active]:text-green-400 text-green-300/70 font-mono">
                <Brain className="h-4 w-4 mr-2" />
                AI_MODELS
              </TabsTrigger>
              <TabsTrigger value="agents" className="data-[state=active]:bg-cyan-900/50 data-[state=active]:text-cyan-400 text-cyan-300/70 font-mono">
                <Bot className="h-4 w-4 mr-2" />
                AGENTS
              </TabsTrigger>
              <TabsTrigger value="execution" className="data-[state=active]:bg-purple-900/50 data-[state=active]:text-purple-400 text-purple-300/70 font-mono">
                <Zap className="h-4 w-4 mr-2" />
                EXECUTION
              </TabsTrigger>
              <TabsTrigger value="risk" className="data-[state=active]:bg-yellow-900/50 data-[state=active]:text-yellow-400 text-yellow-300/70 font-mono">
                <AlertTriangle className="h-4 w-4 mr-2" />
                RISK
              </TabsTrigger>
              <TabsTrigger value="mt5" className="data-[state=active]:bg-blue-900/50 data-[state=active]:text-blue-400 text-blue-300/70 font-mono">
                <Network className="h-4 w-4 mr-2" />
                MT5
              </TabsTrigger>
              <TabsTrigger value="system" className="data-[state=active]:bg-red-900/50 data-[state=active]:text-red-400 text-red-300/70 font-mono">
                <Cpu className="h-4 w-4 mr-2" />
                SYSTEM
              </TabsTrigger>
            </TabsList>

            {/* AI Models Management */}
            <TabsContent value="models" className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-green-400 font-mono">► AI_MODELS_CONFIGURATION</h3>
                <div className="flex space-x-2">
                  <Button
                    onClick={() => handleModelRetrain()}
                    disabled={isTraining}
                    className="flex items-center space-x-2 bg-green-900/30 border-green-500/50 text-green-400 hover:bg-green-800/30 font-mono"
                  >
                    {isTraining ? <Loader className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    <span>RETRAIN_ALL</span>
                  </Button>
                </div>
              </div>

              {isTraining && (
                <Card className="bg-black/50 border-green-500/30">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-green-400 font-mono">TRAINING_PROGRESS</span>
                      <span className="text-sm text-green-300/70 font-mono">{Math.round(trainingProgress)}%</span>
                    </div>
                    <Progress value={trainingProgress} className="w-full" />
                  </CardContent>
                </Card>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {models.map((model, index) => (
                  <Card key={index} className="bg-black/50 border-green-500/30 shadow-lg shadow-green-500/10">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base text-green-400 font-mono">{model.name.toUpperCase()}</CardTitle>
                        <Badge variant={model.status === 'active' ? 'default' : 'secondary'} 
                               className={`font-mono ${model.status === 'active' ? 'bg-green-900/50 border-green-500/50 text-green-400' : 'bg-gray-800/50 border-gray-500/50 text-gray-400'}`}>
                          {model.status.toUpperCase()}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-green-300/70 font-mono">ACCURACY:</span>
                          <div className="font-medium text-green-400 font-mono">{model.accuracy}%</div>
                        </div>
                        <div>
                          <span className="text-green-300/70 font-mono">VERSION:</span>
                          <div className="font-medium text-green-400 font-mono">{model.version}</div>
                        </div>
                        <div>
                          <span className="text-green-300/70 font-mono">SIZE:</span>
                          <div className="font-medium text-green-400 font-mono">{model.size}</div>
                        </div>
                        <div>
                          <span className="text-green-300/70 font-mono">LAST_TRAINED:</span>
                          <div className="font-medium text-green-400 font-mono">{model.last_trained}</div>
                        </div>
                      </div>
                      
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleModelRetrain(model.name)}
                          disabled={isTraining}
                          className="bg-green-900/30 border-green-500/50 text-green-400 hover:bg-green-800/30 font-mono"
                        >
                          <RefreshCw className="h-4 w-4 mr-1" />
                          RETRAIN
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {/* Download model logic */}}
                          className="bg-cyan-900/30 border-cyan-500/50 text-cyan-400 hover:bg-cyan-800/30 font-mono"
                        >
                          <Download className="h-4 w-4 mr-1" />
                          DOWNLOAD
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Workflow Agents Management */}
            <TabsContent value="agents" className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-cyan-400 font-mono">► WORKFLOW_AGENTS_MANAGEMENT</h3>
                <Button variant="outline" className="bg-cyan-900/30 border-cyan-500/50 text-cyan-400 hover:bg-cyan-800/30 font-mono">
                  <Plus className="h-4 w-4 mr-2" />
                  ADD_AGENT
                </Button>
              </div>

              <div className="grid grid-cols-1 gap-4">
                {workflows.map((agent, index) => (
                  <Card key={index} className="bg-black/50 border-cyan-500/30 shadow-lg shadow-cyan-500/10">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base text-cyan-400 font-mono">{agent.name.toUpperCase()}</CardTitle>
                        <div className="flex items-center space-x-2">
                          <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}
                                 className={`font-mono ${agent.status === 'active' ? 'bg-cyan-900/50 border-cyan-500/50 text-cyan-400' : 'bg-gray-800/50 border-gray-500/50 text-gray-400'}`}>
                            {agent.status.toUpperCase()}
                          </Badge>
                          <Switch
                            checked={agent.status === 'active'}
                            onCheckedChange={(checked) => handleWorkflowToggle(agent.name, checked)}
                          />
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-cyan-300/70 font-mono">SUCCESS_RATE:</span>
                          <div className="font-medium text-cyan-400 font-mono">{agent.success_rate}%</div>
                        </div>
                        <div>
                          <span className="text-cyan-300/70 font-mono">LAST_EXECUTION:</span>
                          <div className="font-medium text-cyan-400 font-mono">{agent.last_execution}</div>
                        </div>
                        <div>
                          <span className="text-cyan-300/70 font-mono">STATUS:</span>
                          <div className="font-medium text-cyan-400 font-mono">{agent.status.toUpperCase()}</div>
                        </div>
                      </div>
                      
                      <div className="flex space-x-2 mt-4">
                        <Button size="sm" variant="outline" className="bg-cyan-900/30 border-cyan-500/50 text-cyan-400 hover:bg-cyan-800/30 font-mono">
                          <Activity className="h-3 w-3 mr-1" />
                          LOGS
                        </Button>
                        <Button size="sm" variant="outline" className="bg-purple-900/30 border-purple-500/50 text-purple-400 hover:bg-purple-800/30 font-mono">
                          <BarChart3 className="h-3 w-3 mr-1" />
                          ANALYTICS
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Execution Engine Configuration */}
            <TabsContent value="execution" className="space-y-6">
              <h3 className="text-lg font-semibold text-purple-400 font-mono">► EXECUTION_ENGINE_CONFIGURATION</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="bg-black/50 border-purple-500/30 shadow-lg shadow-purple-500/10">
                  <CardHeader>
                    <CardTitle className="text-base text-purple-400 font-mono">ORDER_MANAGEMENT</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-purple-300/70 font-mono">MAX_POSITION_SIZE</Label>
                      <Input type="number" className="bg-black/50 border-purple-500/50 text-purple-400 font-mono" />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-purple-300/70 font-mono">ORDER_TIMEOUT (seconds)</Label>
                      <Input type="number" className="bg-black/50 border-purple-500/50 text-purple-400 font-mono" />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-purple-300/70 font-mono">SLIPPAGE_TOLERANCE</Label>
                      <Select>
                        <SelectTrigger className="bg-black/50 border-purple-500/50 text-purple-400 font-mono">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-black border-purple-500/50">
                          <SelectItem value="low" className="text-purple-400 font-mono">LOW (1 pip)</SelectItem>
                          <SelectItem value="medium" className="text-purple-400 font-mono">MEDIUM (3 pips)</SelectItem>
                          <SelectItem value="high" className="text-purple-400 font-mono">HIGH (5 pips)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base text-purple-400 font-mono">EXECUTION_STRATEGY</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-purple-300/70 font-mono">EXECUTION_ALGORITHM</Label>
                      <Select>
                        <SelectTrigger className="bg-black/50 border-purple-500/50 text-purple-400 font-mono">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-black border-purple-500/50">
                          <SelectItem value="market" className="text-purple-400 font-mono">Market</SelectItem>
                          <SelectItem value="twap" className="text-purple-400 font-mono">TWAP</SelectItem>
                          <SelectItem value="vwap" className="text-purple-400 font-mono">VWAP</SelectItem>
                          <SelectItem value="iceberg" className="text-purple-400 font-mono">Iceberg</SelectItem>
                          <SelectItem value="iceberg">Iceberg</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Enable Smart Routing</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Pre-Trade Risk Checks</Label>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button onClick={() => handleConfigUpdate('execution', {})}>
                <Save className="h-4 w-4 mr-2" />
                Save Configuration
              </Button>
            </TabsContent>

            {/* Risk Management */}
            <TabsContent value="risk" className="space-y-6">
              <h3 className="text-lg font-semibold">Risk Management Configuration</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Position Limits</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Max Daily Loss (%)</Label>
                      <Input />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Positions</Label>
                      <Input />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Leverage</Label>
                      <Input />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Risk Controls</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Enable Kill Switch</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Auto Stop Loss</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Correlation Limits</Label>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button onClick={() => handleConfigUpdate('risk', {})}>
                <Save className="h-4 w-4 mr-2" />
                Save Configuration
              </Button>
            </TabsContent>

            {/* MT5 Configuration */}
            <TabsContent value="mt5" className="space-y-6">
              <h3 className="text-lg font-semibold">MT5 Connection Configuration</h3>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Connection Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Server</Label>
                      <Input />
                    </div>
                    <div className="space-y-2">
                      <Label>Login</Label>
                      <Input />
                    </div>
                    <div className="space-y-2">
                      <Label>Password</Label>
                      <Input type="password" />
                    </div>
                    <div className="space-y-2">
                      <Label>Connection Timeout (ms)</Label>
                      <Input />
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Switch />
                    <Label>Enable Real-time Data Feed</Label>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button variant="outline">
                      <Activity className="h-4 w-4 mr-2" />
                      Test Connection
                    </Button>
                    <Button onClick={() => handleConfigUpdate('mt5', {})}>
                      <Save className="h-4 w-4 mr-2" />
                      Save Configuration
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* System Configuration */}
            <TabsContent value="system" className="space-y-6">
              <h3 className="text-lg font-semibold">System Configuration</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Performance Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>CPU Cores</Label>
                      <Select>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto</SelectItem>
                          <SelectItem value="1">1</SelectItem>
                          <SelectItem value="2">2</SelectItem>
                          <SelectItem value="4">4</SelectItem>
                          <SelectItem value="8">8</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Memory Allocation (GB)</Label>
                      <Input />
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Enable GPU Acceleration</Label>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Logging & Monitoring</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Log Level</Label>
                      <Select>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="debug">DEBUG</SelectItem>
                          <SelectItem value="info">INFO</SelectItem>
                          <SelectItem value="warning">WARNING</SelectItem>
                          <SelectItem value="error">ERROR</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Enable Metrics Collection</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch />
                      <Label>Real-time Alerts</Label>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button onClick={() => handleConfigUpdate('system', {})} className="bg-red-900/30 border-red-500/50 text-red-400 hover:bg-red-800/30 font-mono">
                <Save className="h-4 w-4 mr-2" />
                SAVE_CONFIGURATION
              </Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
