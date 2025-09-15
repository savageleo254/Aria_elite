import { DatabaseService } from '../src/lib/db'

async function initializeDatabase() {
  try {
    console.log('Initializing ARIA ELITE database...')

    // Create sample symbols
    const eurusd = await DatabaseService.createSymbol({
      name: 'EURUSD',
      displayName: 'EUR/USD',
      category: 'forex'
    })

    const gbpusd = await DatabaseService.createSymbol({
      name: 'GBPUSD',
      displayName: 'GBP/USD',
      category: 'forex'
    })

    const usdjpy = await DatabaseService.createSymbol({
      name: 'USDJPY',
      displayName: 'USD/JPY',
      category: 'forex'
    })

    console.log('Created symbols:', { eurusd: eurusd.id, gbpusd: gbpusd.id, usdjpy: usdjpy.id })

    // Create sample signals
    const signal1 = await DatabaseService.createSignal({
      symbolId: eurusd.id,
      direction: 'buy',
      entryPrice: 1.0850,
      stopLoss: 1.0800,
      takeProfit: 1.0950,
      confidence: 0.85,
      strategy: 'smc',
      timeframe: '1h',
      aiConfidence: 0.82,
      smcScore: 0.88,
      parameters: JSON.stringify({ min_confidence: 0.75, max_risk_per_trade: 0.02 })
    })

    const signal2 = await DatabaseService.createSignal({
      symbolId: gbpusd.id,
      direction: 'sell',
      entryPrice: 1.2650,
      stopLoss: 1.2700,
      takeProfit: 1.2550,
      confidence: 0.78,
      strategy: 'ai',
      timeframe: '4h',
      aiConfidence: 0.75,
      parameters: JSON.stringify({ ensemble_threshold: 0.7 })
    })

    console.log('Created signals:', { signal1: signal1.id, signal2: signal2.id })

    // Create sample trades
    const trade1 = await DatabaseService.createTrade({
      symbolId: eurusd.id,
      signalId: signal1.id,
      direction: 'buy',
      volume: 0.1,
      entryPrice: 1.0850,
      stopLoss: 1.0800,
      takeProfit: 1.0950
    })

    const trade2 = await DatabaseService.createTrade({
      symbolId: gbpusd.id,
      signalId: signal2.id,
      direction: 'sell',
      volume: 0.05,
      entryPrice: 1.2650,
      stopLoss: 1.2700,
      takeProfit: 1.2550
    })

    console.log('Created trades:', { trade1: trade1.id, trade2: trade2.id })

    // Add some market data
    const now = new Date()
    for (let i = 0; i < 100; i++) {
      const timestamp = new Date(now.getTime() - i * 60000) // 1 minute intervals
      const basePrice = 1.0850 + (Math.random() - 0.5) * 0.01
      
      await DatabaseService.addMarketData({
        symbolId: eurusd.id,
        timestamp,
        open: basePrice,
        high: basePrice + Math.random() * 0.005,
        low: basePrice - Math.random() * 0.005,
        close: basePrice + (Math.random() - 0.5) * 0.002,
        volume: Math.random() * 1000,
        sma20: basePrice + (Math.random() - 0.5) * 0.001,
        sma50: basePrice + (Math.random() - 0.5) * 0.002,
        rsi14: 30 + Math.random() * 40,
        atr: 0.001 + Math.random() * 0.002
      })
    }

    console.log('Added market data for EURUSD')

    // Create AI models
    const lstmModel = await DatabaseService.createAIModel({
      name: 'LSTM_Price_Predictor',
      type: 'lstm',
      version: '1.0.0',
      status: 'ready',
      accuracy: 0.78,
      loss: 0.15,
      parameters: JSON.stringify({ 
        sequence_length: 60, 
        hidden_units: 128, 
        dropout: 0.2 
      })
    })

    const lightgbmModel = await DatabaseService.createAIModel({
      name: 'LightGBM_Classifier',
      type: 'lightgbm',
      version: '1.0.0',
      status: 'ready',
      accuracy: 0.82,
      loss: 0.12,
      parameters: JSON.stringify({ 
        n_estimators: 100, 
        learning_rate: 0.1, 
        max_depth: 6 
      })
    })

    console.log('Created AI models:', { lstm: lstmModel.id, lightgbm: lightgbmModel.id })

    // Add performance metrics
    await DatabaseService.addPerformanceMetrics({
      date: new Date(),
      totalTrades: 2,
      winningTrades: 1,
      losingTrades: 1,
      winRate: 0.5,
      totalProfit: 25.0,
      totalLoss: 12.5,
      netProfit: 12.5,
      maxDrawdown: 0.05,
      sharpeRatio: 1.2,
      profitFactor: 2.0,
      activeSignals: 2,
      activeTrades: 2
    })

    console.log('Added performance metrics')

    // Log initialization
    await DatabaseService.log('info', 'system', 'Database initialized with sample data')

    console.log('✅ Database initialization completed successfully!')
    
  } catch (error) {
    console.error('❌ Database initialization failed:', error)
    throw error
  }
}

// Run the initialization
initializeDatabase()
  .then(() => {
    console.log('Database initialization script completed')
    process.exit(0)
  })
  .catch((error) => {
    console.error('Database initialization script failed:', error)
    process.exit(1)
  })
