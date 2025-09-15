"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Zap, Clock, CheckCircle, AlertCircle, XCircle } from 'lucide-react'

interface AIModelConfig {
  provider: 'openai' | 'google' | 'xai' | 'anthropic' | 'deepseek' | 'perplexity';
  model: string;
  account: string;
  status: 'active' | 'rate_limited' | 'error' | 'maintenance';
  dailyUsage: number;
  dailyLimit: number;
  hourlyUsage: number;
  hourlyLimit: number;
  lastUsed: Date;
  avgResponseTime: number;
  successRate: number;
  priority: number;
  tier: 'critical' | 'complex' | 'standard' | 'efficient';
  subscription: 'premium' | 'pro' | 'standard' | 'free';
}

interface ModelProvider {
  id: string;
  name: string;
  icon: string;
  status: 'connected' | 'disconnected' | 'error';
  models: AIModelConfig[];
}

const ModelSelectionTab: React.FC = () => {
  const [providers, setProviders] = useState<ModelProvider[]>([])
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedAccount, setSelectedAccount] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [autoRotation, setAutoRotation] = useState(true)

  // Sample data - would be loaded from API
  const sampleProviders: ModelProvider[] = [
    {
      id: 'openai',
      name: 'OpenAI',
      icon: '🤖',
      status: 'connected',
      models: [
        {
          provider: 'openai',
          model: 'gpt-5',
          account: 'premium-account-1',
          status: 'active',
          dailyUsage: 1,
          dailyLimit: 2,
          hourlyUsage: 0,
          hourlyLimit: 1,
          lastUsed: new Date(Date.now() - 30 * 60000),
          avgResponseTime: 2.3,
          successRate: 0.98,
          priority: 10,
          tier: 'critical',
          subscription: 'premium'
        },
        {
          provider: 'openai',
          model: 'gpt-4-turbo-2024',
          account: 'standard-account-1',
          status: 'active',
          dailyUsage: 5,
          dailyLimit: 10,
          hourlyUsage: 2,
          hourlyLimit: 3,
          lastUsed: new Date(Date.now() - 15 * 60000),
          avgResponseTime: 1.8,
          successRate: 0.96,
          priority: 8,
          tier: 'complex',
          subscription: 'pro'
        }
      ]
    },
    {
      id: 'google',
      name: 'Google Gemini',
      icon: '💎',
      status: 'connected',
      models: [
        {
          provider: 'google',
          model: 'gemini-2.5-pro',
          account: 'premium-account-1',
          status: 'active',
          dailyUsage: 2,
          dailyLimit: 5,
          hourlyUsage: 1,
          hourlyLimit: 2,
          lastUsed: new Date(Date.now() - 45 * 60000),
          avgResponseTime: 2.1,
          successRate: 0.97,
          priority: 9,
          tier: 'critical',
          subscription: 'premium'
        }
      ]
    },
    {
      id: 'xai',
      name: 'xAI Grok',
      icon: '⚡',
      status: 'connected',
      models: [
        {
          provider: 'xai',
          model: 'grok-4',
          account: 'premium-account-1',
          status: 'rate_limited',
          dailyUsage: 8,
          dailyLimit: 10,
          hourlyUsage: 3,
          hourlyLimit: 3,
          lastUsed: new Date(Date.now() - 10 * 60000),
          avgResponseTime: 1.5,
          successRate: 0.94,
          priority: 7,
          tier: 'complex',
          subscription: 'premium'
        }
      ]
    }
  ]

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      setProviders(sampleProviders)
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'rate_limited':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'maintenance':
        return <AlertCircle className="h-4 w-4 text-blue-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500'
      case 'rate_limited':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-500'
      case 'maintenance':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'critical':
        return 'bg-red-100 text-red-800'
      case 'complex':
        return 'bg-orange-100 text-orange-800'
      case 'standard':
        return 'bg-blue-100 text-blue-800'
      case 'efficient':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const selectedProviderData = providers.find(p => p.id === selectedProvider)
  const availableModels = selectedProviderData?.models || []
  const selectedModelData = availableModels.find(m => m.model === selectedModel)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Model Selection</h1>
          <p className="text-muted-foreground">Premium browser automation - No API keys required</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={autoRotation ? "default" : "secondary"}>
            Auto Rotation: {autoRotation ? "ON" : "OFF"}
          </Badge>
          <Button
            variant="outline"
            onClick={() => setAutoRotation(!autoRotation)}
          >
            {autoRotation ? "Disable" : "Enable"} Auto Rotation
          </Button>
        </div>
      </div>

      <Tabs defaultValue="selection" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="selection">Model Selection</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="accounts">Account Management</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="selection" className="space-y-6">
          {/* Provider Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Provider Selection
              </CardTitle>
              <CardDescription>
                Choose AI provider and model for your trading signals
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {providers.map((provider) => (
                  <Card
                    key={provider.id}
                    className={`cursor-pointer transition-all ${
                      selectedProvider === provider.id
                        ? 'ring-2 ring-blue-500'
                        : 'hover:shadow-md'
                    }`}
                    onClick={() => setSelectedProvider(provider.id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="text-2xl">{provider.icon}</span>
                          <span className="font-semibold">{provider.name}</span>
                        </div>
                        <div className={`w-2 h-2 rounded-full ${getStatusColor(provider.status)}`} />
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {provider.models.length} models available
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Model Selection */}
          {selectedProviderData && (
            <Card>
              <CardHeader>
                <CardTitle>Available Models - {selectedProviderData.name}</CardTitle>
                <CardDescription>
                  Select model based on task complexity and availability
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {availableModels.map((model) => (
                    <Card
                      key={model.model}
                      className={`cursor-pointer transition-all ${
                        selectedModel === model.model
                          ? 'ring-2 ring-blue-500'
                          : 'hover:shadow-sm'
                      }`}
                      onClick={() => setSelectedModel(model.model)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(model.status)}
                              <span className="font-semibold">{model.model}</span>
                            </div>
                            <Badge className={getTierColor(model.tier)}>
                              {model.tier}
                            </Badge>
                            <Badge variant="outline">
                              {model.subscription}
                            </Badge>
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {model.avgResponseTime}s avg
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <div className="text-muted-foreground">Daily Usage</div>
                            <div className="flex items-center space-x-2">
                              <Progress
                                value={(model.dailyUsage / model.dailyLimit) * 100}
                                className="flex-1"
                              />
                              <span>{model.dailyUsage}/{model.dailyLimit}</span>
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Hourly Usage</div>
                            <div className="flex items-center space-x-2">
                              <Progress
                                value={(model.hourlyUsage / model.hourlyLimit) * 100}
                                className="flex-1"
                              />
                              <span>{model.hourlyUsage}/{model.hourlyLimit}</span>
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Success Rate</div>
                            <div className="font-semibold text-green-600">
                              {(model.successRate * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Last Used</div>
                            <div className="text-xs">
                              {Math.floor((Date.now() - model.lastUsed.getTime()) / 60000)}m ago
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Account Selection */}
          {selectedModelData && (
            <Card>
              <CardHeader>
                <CardTitle>Account Configuration</CardTitle>
                <CardDescription>
                  Browser automation account for {selectedModelData.model}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                  <div>
                    <div className="font-semibold">Account: {selectedModelData.account}</div>
                    <div className="text-sm text-muted-foreground">
                      Browser session active - No API key required
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(selectedModelData.status)}
                    <span className="text-sm capitalize">{selectedModelData.status}</span>
                  </div>
                </div>
                
                <div className="flex space-x-4">
                  <Button>
                    <Zap className="h-4 w-4 mr-2" />
                    Test Connection
                  </Button>
                  <Button variant="outline">
                    Refresh Session
                  </Button>
                  <Button variant="outline">
                    View Logs
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>
                Real-time performance data for all models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {providers.flatMap(p => p.models).map((model, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-muted rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div>
                        <div className="font-semibold">{model.model}</div>
                        <div className="text-sm text-muted-foreground">{model.provider}</div>
                      </div>
                      <Badge className={getTierColor(model.tier)}>
                        {model.tier}
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-6 text-sm">
                      <div className="text-center">
                        <div className="text-muted-foreground">Response Time</div>
                        <div className="font-semibold">{model.avgResponseTime}s</div>
                      </div>
                      <div className="text-center">
                        <div className="text-muted-foreground">Success Rate</div>
                        <div className="font-semibold text-green-600">
                          {(model.successRate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-muted-foreground">Priority</div>
                        <div className="font-semibold">{model.priority}/10</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="accounts">
          <Card>
            <CardHeader>
              <CardTitle>Account Management</CardTitle>
              <CardDescription>
                Manage browser automation accounts and credentials
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                Account management interface - Configure browser automation credentials
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings">
          <Card>
            <CardHeader>
              <CardTitle>Model Settings</CardTitle>
              <CardDescription>
                Configure model selection and rotation settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                Settings interface - Configure rotation strategies and preferences
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default ModelSelectionTab
