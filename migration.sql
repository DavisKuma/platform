-- Adzuna Sponsorship Jobs Scraper — Supabase Migration
-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- Table: jobs (scraped sponsored jobs from Adzuna)
CREATE TABLE IF NOT EXISTS jobs (
  id BIGSERIAL PRIMARY KEY,
  job_title TEXT,
  company TEXT,
  location TEXT,
  category TEXT,
  salary TEXT,
  contract_type TEXT,
  contract_time TEXT,
  posted TIMESTAMPTZ,
  match_type TEXT,
  url TEXT UNIQUE,
  description TEXT,
  scraped_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: recommendations (CV-matched jobs via OpenAI)
CREATE TABLE IF NOT EXISTS recommendations (
  id BIGSERIAL PRIMARY KEY,
  cv_filename TEXT,
  relevance_score INTEGER,
  reason TEXT,
  job_title TEXT,
  company TEXT,
  location TEXT,
  category TEXT,
  salary TEXT,
  contract_type TEXT,
  contract_time TEXT,
  posted TEXT,
  url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS policies (allow full access for anon key — personal tool)
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on jobs" ON jobs FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on recommendations" ON recommendations FOR ALL USING (true) WITH CHECK (true);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_scraped_at ON jobs(scraped_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_cv ON recommendations(cv_filename);
CREATE INDEX IF NOT EXISTS idx_recommendations_score ON recommendations(relevance_score DESC);
