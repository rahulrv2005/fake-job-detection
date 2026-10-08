/**
 * SecureHire ML - Express Server Entry Point
 * Implements full REST API and mounts Vite middleware on port 3000.
 */

import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { MLPipeline } from './ml/trainer.ts';
import { DatabaseManager } from './database/db.ts';
import { RAW_DATASET } from './dataset/seed_data.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isProduction = process.env.NODE_ENV === 'production';
const PORT = 3000;

async function bootstrapServer() {
  const app = express();
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true }));

  // Initialize SQLite Database
  const dbManager = DatabaseManager.getInstance();
  await dbManager.init();

  // Initialize and train ML Pipeline on dataset
  const pipeline = MLPipeline.getInstance();
  console.log('[Server] Starting initial ML training on dataset...');
  await pipeline.train();

  // -------------------------------------------------------------
  // REST API Endpoints
  // -------------------------------------------------------------

  /**
   * POST /api/predict
   * Evaluates job posting using TF-IDF + Best ML model (or specified algorithm)
   * Saves result to SQLite database
   */
  app.post('/api/predict', async (req: Request, res: Response) => {
    try {
      const jobInput = req.body;
      if (!jobInput || (!jobInput.title && !jobInput.description)) {
        return res.status(400).json({ error: 'Please provide at least a Job Title or Job Description.' });
      }

      const requestedModel = req.query.model as string | undefined;
      const predictionResult = pipeline.predict(jobInput, requestedModel);

      // Save to SQLite Database
      const savedRecord = dbManager.insertPrediction({
        job_title: predictionResult.jobTitle,
        company_name: predictionResult.companyName,
        location: jobInput.location || 'Remote / Unspecified',
        prediction: predictionResult.prediction,
        confidence: predictionResult.confidence,
        model_used: predictionResult.modelUsed,
        indicators: predictionResult.indicators.map(i => i.title)
      });

      return res.json({
        success: true,
        recordId: savedRecord.id,
        ...predictionResult
      });
    } catch (err: any) {
      console.error('[API] Prediction error:', err);
      return res.status(500).json({ error: err.message || 'Internal error during classification' });
    }
  });

  /**
   * GET /api/model-comparison
   * Returns evaluation metrics for all 5 algorithms
   */
  app.get('/api/model-comparison', (req: Request, res: Response) => {
    try {
      const summary = pipeline.trainingSummary;
      if (!summary) {
        return res.status(503).json({ error: 'Models are still training. Please retry shortly.' });
      }
      return res.json(summary);
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * POST /api/retrain
   * Retrains all 5 models and returns updated performance metrics
   */
  app.post('/api/retrain', async (req: Request, res: Response) => {
    try {
      console.log('[Server] Retraining models requested by user...');
      const summary = await pipeline.train();
      return res.json({
        success: true,
        message: 'All 5 algorithms retrained and evaluated successfully.',
        summary
      });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * GET /api/dashboard
   * Computes high-level analytics from SQLite database
   */
  app.get('/api/dashboard', (req: Request, res: Response) => {
    try {
      const bestAlg = pipeline.trainingSummary?.bestAlgorithm || 'Random Forest';
      const stats = dbManager.getDashboardStats(bestAlg);
      return res.json(stats);
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * GET /api/history
   * Retrieves prediction log from SQLite
   */
  app.get('/api/history', (req: Request, res: Response) => {
    try {
      const history = dbManager.getHistory(100);
      return res.json(history);
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * DELETE /api/history/:id
   * Deletes a record from SQLite
   */
  app.delete('/api/history/:id', (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id, 10);
      const success = dbManager.deletePrediction(id);
      return res.json({ success });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * DELETE /api/history
   * Clears all predictions from SQLite
   */
  app.delete('/api/history', (req: Request, res: Response) => {
    try {
      const success = dbManager.clearAllPredictions();
      return res.json({ success });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  /**
   * GET /api/samples
   * Pre-packaged real and fraudulent sample postings for quick test demo
   */
  app.get('/api/samples', (req: Request, res: Response) => {
    const samples = [
      {
        id: 'real-stripe',
        tag: 'Legitimate Tech Role',
        isFakeExpected: false,
        data: RAW_DATASET[0]
      },
      {
        id: 'real-nurse',
        tag: 'Legitimate Healthcare Role',
        isFakeExpected: false,
        data: RAW_DATASET[1]
      },
      {
        id: 'fake-check-scam',
        tag: 'Scam: Cashier Check & Telegram',
        isFakeExpected: true,
        data: RAW_DATASET[15] // Apex Global Data Entry
      },
      {
        id: 'fake-wire-mule',
        tag: 'Scam: Wire Transfer Mule',
        isFakeExpected: true,
        data: RAW_DATASET[16] // Swift International Wire Rep
      },
      {
        id: 'fake-fee-scam',
        tag: 'Scam: Upfront Registration Fee',
        isFakeExpected: true,
        data: RAW_DATASET[18] // Prime Horizons Assistant
      }
    ];
    return res.json(samples);
  });

  // -------------------------------------------------------------
  // Frontend Serving (Vite middleware in dev, dist static in prod)
  // -------------------------------------------------------------
  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SecureHire ML] Server running on http://0.0.0.0:${PORT}`);
  });
}

bootstrapServer().catch(err => {
  console.error('[SecureHire ML] Fatal bootstrap error:', err);
  process.exit(1);
});
