-- Supabase schema for MicroSponsor AI
-- Run this in your Supabase SQL Editor

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    subscription_tier TEXT DEFAULT 'free',
    daily_email_count INTEGER DEFAULT 0,
    daily_email_reset_date DATE DEFAULT CURRENT_DATE,
    cv_url TEXT,
    cv_filename TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Jobs table
CREATE TABLE IF NOT EXISTS public.jobs (
    id SERIAL PRIMARY KEY,
    job_title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT DEFAULT '',
    category TEXT DEFAULT '',
    salary TEXT DEFAULT '',
    salary_min INTEGER DEFAULT 0,
    salary_max INTEGER DEFAULT 0,
    contract_type TEXT DEFAULT '',
    contract_time TEXT DEFAULT '',
    posted TEXT DEFAULT '',
    match_type TEXT DEFAULT '',
    url TEXT UNIQUE,
    description TEXT DEFAULT '',
    source TEXT DEFAULT 'adzuna',
    industry TEXT DEFAULT 'General',
    employee_count INTEGER DEFAULT 0,
    is_new_entrant_friendly BOOLEAN DEFAULT TRUE,
    is_verified_sponsor BOOLEAN DEFAULT FALSE,
    hiring_manager TEXT DEFAULT '',
    hiring_manager_email TEXT DEFAULT '',
    company_description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recommendations table
CREATE TABLE IF NOT EXISTS public.recommendations (
    id SERIAL PRIMARY KEY,
    cv_filename TEXT,
    relevance_score INTEGER DEFAULT 0,
    reason TEXT DEFAULT '',
    job_title TEXT DEFAULT '',
    company TEXT DEFAULT '',
    location TEXT DEFAULT '',
    category TEXT DEFAULT '',
    salary TEXT DEFAULT '',
    contract_type TEXT DEFAULT '',
    contract_time TEXT DEFAULT '',
    posted TEXT DEFAULT '',
    url TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Saved matches
CREATE TABLE IF NOT EXISTS public.saved_matches (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES public.jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, job_id)
);

-- Outreach history
CREATE TABLE IF NOT EXISTS public.outreach_history (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES public.jobs(id) ON DELETE SET NULL,
    company TEXT NOT NULL,
    hiring_manager TEXT DEFAULT '',
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT DEFAULT 'generated',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sponsors table
CREATE TABLE IF NOT EXISTS public.sponsors (
    id SERIAL PRIMARY KEY,
    organisation_name TEXT NOT NULL,
    normalised_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sponsors_normalised ON public.sponsors(normalised_name);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.saved_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.outreach_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sponsors ENABLE ROW LEVEL SECURITY;

-- Public read access for jobs and sponsors
CREATE POLICY IF NOT EXISTS "Public read jobs" ON public.jobs FOR SELECT USING (true);
CREATE POLICY IF NOT EXISTS "Public read sponsors" ON public.sponsors FOR SELECT USING (true);

-- Service role can do everything (Python API uses service key)
CREATE POLICY IF NOT EXISTS "Service full access jobs" ON public.jobs FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "Service full access users" ON public.users FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "Service full access saved_matches" ON public.saved_matches FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "Service full access outreach" ON public.outreach_history FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "Service full access sponsors" ON public.sponsors FOR ALL USING (true);
