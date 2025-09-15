"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Send, Bot, User, Settings, Download, Trash2, RefreshCw, Zap } from 'lucide-react'
import { Textarea } from "@/components/ui/textarea"
import styles from './ModelChatInterface.module.css'

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model: string;
  responseTime?: number;
  tokenCount?: number;
}

interface ModelCapabilities {
  maxTokens: number;
  supportsStreaming: boolean;
  supportsFiles: boolean;
  specialFeatures: string[];
}

interface ModelChatProps {
  selectedModel?: {
    provider: string;
    model: string;
    account: string;
    status: string;
    tier: string;
  };
}

const ModelChatInterface: React.FC<ModelChatProps> = ({ selectedModel }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [temperature, setTemperature] = useState<number[]>([0.7])
  const [maxTokens, setMaxTokens] = useState<number[]>([2000])
  const [showSettings, setShowSettings] = useState(false)
  const [responseStats, setResponseStats] = useState({
    avgResponseTime: 0,
    totalTokens: 0,
    totalMessages: 0
  })
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const modelCapabilities: ModelCapabilities = {
    maxTokens: selectedModel?.model === 'gpt-5' ? 8000 : 
               selectedModel?.model === 'gemini-2.5-pro' ? 8000 :
               selectedModel?.model === 'grok-4' ? 6000 : 4000,
    supportsStreaming: true,
    supportsFiles: selectedModel?.tier === 'critical',
    specialFeatures: selectedModel?.model === 'gpt-5' ? ['Advanced Reasoning', 'Code Generation', 'Image Analysis'] :
                    selectedModel?.model === 'gemini-2.5-pro' ? ['Multimodal', 'Long Context', 'Code Generation'] :
                    selectedModel?.model === 'grok-4' ? ['Real-time Data', 'X Integration', 'Humor'] :
                    ['Standard Chat', 'Text Generation']
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedModel) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
      model: selectedModel.model
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsTyping(true)

    try {
      // Simulate API call to backend browser automation
      const startTime = Date.now()
      
      const response = await fetch('/api/chat/premium-models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          model: selectedModel.model,
          provider: selectedModel.provider,
          temperature: temperature[0],
          maxTokens: maxTokens[0]
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      const responseTime = Date.now() - startTime

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.content || 'Sorry, I couldn\'t generate a response. The browser automation might need attention.',
        timestamp: new Date(),
        model: selectedModel.model,
        responseTime: responseTime / 1000,
        tokenCount: data.tokenCount || 0
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Update stats
      setResponseStats(prev => ({
        avgResponseTime: (prev.avgResponseTime * prev.totalMessages + responseTime / 1000) / (prev.totalMessages + 1),
        totalTokens: prev.totalTokens + (data.tokenCount || 0),
        totalMessages: prev.totalMessages + 1
      }))

    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Error: Browser automation connection failed. Please check model status and try again.',
        timestamp: new Date(),
        model: selectedModel.model
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
    setResponseStats({
      avgResponseTime: 0,
      totalTokens: 0,
      totalMessages: 0
    })
  }

  const exportChat = () => {
    const chatData = {
      model: selectedModel?.model,
      provider: selectedModel?.provider,
      timestamp: new Date().toISOString(),
      messages: messages,
      stats: responseStats
    }
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chat-${selectedModel?.model}-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const testConnection = async () => {
    setIsTyping(true)
    try {
      const response = await fetch('/api/chat/test-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel?.model,
          provider: selectedModel?.provider
        })
      })
      
      const data = await response.json()
      
      const testMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.success ? 
          `✅ Connection test successful for ${selectedModel?.model}. Browser session is active and ready.` :
          `❌ Connection test failed for ${selectedModel?.model}. Error: ${data.error}`,
        timestamp: new Date(),
        model: selectedModel?.model || 'system'
      }
      
      setMessages(prev => [...prev, testMessage])
    } catch (error) {
      console.error('Connection test failed:', error)
    } finally {
      setIsTyping(false)
    }
  }

  if (!selectedModel) {
    return (
      <Card className="h-full">
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center text-muted-foreground">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Select a model from the Model Selection tab to start chatting</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5" />
                {selectedModel.model}
              </CardTitle>
              <CardDescription>
                {selectedModel.provider} • {selectedModel.tier} tier • Browser automation active
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant={selectedModel.status === 'active' ? 'default' : 'secondary'}>
                {selectedModel.status}
              </Badge>
              <Button variant="outline" size="sm" onClick={() => setShowSettings(!showSettings)}>
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        {showSettings && (
          <CardContent className="pt-0">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              <div className="space-y-2">
                <label className="text-sm font-medium">Temperature:</label>
                <Slider 
                  value={temperature} 
                  onValueChange={setTemperature}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-32"
                />
                <span className="text-sm w-10">{temperature[0].toFixed(2)}</span>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Tokens:</label>
                <Slider 
                  value={maxTokens} 
                  onValueChange={setMaxTokens}
                  min={100}
                  max={modelCapabilities.maxTokens}
                  step={100}
                  className="w-32"
                />
                <span className="text-sm w-16">{maxTokens[0]}</span>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Info</label>
                <div className="text-xs text-muted-foreground">
                  {modelCapabilities.specialFeatures.slice(0, 2).join(', ')}
                </div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Chat Messages */}
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>Messages: {messages.length}</span>
              <span>Avg Response: {responseStats.avgResponseTime.toFixed(1)}s</span>
              <span>Tokens: {responseStats.totalTokens}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={testConnection}>
                <Zap className="h-4 w-4 mr-1" />
                Test
              </Button>
              <Button variant="outline" size="sm" onClick={exportChat}>
                <Download className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={clearChat}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="flex-1 flex flex-col min-h-0 p-0">
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.length === 0 && (
                <div className="text-center text-muted-foreground py-8">
                  <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Start a conversation with {selectedModel.model}</p>
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2 max-w-md mx-auto">
                    {modelCapabilities.specialFeatures.map((feature, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex items-start space-x-3 ${
                    message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === 'user' ? 'bg-blue-500' : 'bg-green-500'
                  }`}>
                    {message.role === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>
                  
                  <div className={`flex-1 max-w-3xl ${
                    message.role === 'user' ? 'text-right' : ''
                  }`}>
                    <div className={`inline-block p-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-muted'
                    }`}>
                      <div className="whitespace-pre-wrap">{message.content}</div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                      <span>{message.timestamp.toLocaleTimeString()}</span>
                      {message.responseTime && (
                        <span className="flex items-center space-x-2">
                          <span>{message.responseTime.toFixed(1)}s</span>
                          {message.tokenCount && <span>{message.tokenCount} tokens</span>}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                  <div className="bg-muted p-3 rounded-lg">
                    <div className="flex space-x-1">
                      <div className={styles.typingDot} />
                      <div className={styles.typingDot} />
                      <div className={styles.typingDot} />
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
          
          <Separator />
          
          {/* Input Area */}
          <div className="p-4">
            <div className="flex space-x-2">
              <Textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={`Message ${selectedModel.model}...`}
                className="flex-1 min-h-[40px] max-h-32 resize-none"
                disabled={isTyping}
              />
              <Button 
                onClick={handleSendMessage} 
                disabled={!inputMessage.trim() || isTyping}
                className="self-end"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span>
                {inputMessage.length} / {modelCapabilities.maxTokens * 4} chars (approx)
              </span>
              <span>
                Press Enter to send, Shift+Enter for new line
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ModelChatInterface
