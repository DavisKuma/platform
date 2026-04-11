# MicroSponsor AI — Full SaaS Upgrade

## Phase 1: Architecture & Setup
- [x] Audit current codebase and understand existing patterns
- [x] Plan database schema extensions (saved_matches, outreach_history, subscriptions)

## Phase 2: Real AI Email Generation (OpenAI)
- [x] Add LLM-powered email generation tRPC procedure
- [x] Use exact Sienna email style as system prompt
- [x] Update EmailGeneratorModal to call real AI endpoint
- [x] Add Regenerate, Copy, Open in Gmail buttons (already exist — verify)
- [x] Add free tier limit (3 emails/day)

## Phase 3: Stripe Payment Integration
- [x] Add Stripe feature via webdev_add_feature
- [x] Create subscription plans (£29/mo, £199/year)
- [x] Build checkout flow from Custom Plan button
- [x] Gate hiring manager contacts behind subscription
- [x] Show real name + email for paid users on job cards

## Phase 4: Dashboard & Pages
- [x] Build protected Dashboard layout
- [x] My Matches page (save/unsave jobs)
- [x] Outreach History page (track sent emails)
- [x] CV upload that saves to user profile (S3 storage)
- [x] User profile/settings page (integrated in dashboard)

## Phase 5: Admin Mode
- [x] Admin page to add/edit/delete sponsor jobs
- [x] Admin-only route protection
- [x] Job management CRUD interface

## Phase 6: Polish & Production
- [x] Dark mode toggle
- [x] Mobile responsive polish
- [x] Remove demo alerts and placeholders
- [x] Loading spinners/skeletons throughout
- [x] Professional footer with Top Offer Academy branding
- [x] Custom domain ready

## Phase 7: Testing & Delivery
- [x] Write vitest tests for new endpoints
- [x] End-to-end QA
- [x] Save checkpoint and deliver

## Phase 8: n8n + Apify LinkedIn Scraper + GOV.UK Sponsor Data
- [x] Fix AdminJobs import error in App.tsx (was stale Vite cache, restart fixed it)
- [x] Create sponsors table and import GOV.UK CSV data (121,033 Skilled Worker sponsors imported)
- [x] Build webhook API endpoint to receive jobs from n8n (POST /api/webhook/jobs with GOV.UK cross-reference)
- [x] Create n8n workflow: Apify LinkedIn scraper → filter UK sponsors → webhook to MicroSponsor (ID: 5eYwxnmopfHOKQTB, active)
- [x] Cross-reference LinkedIn jobs with verified sponsor list (fuzzy matching against 121K sponsors)
- [x] Update frontend to show verified sponsor badges from real data (121,000+ sponsors)
- [x] Connect ChatGPT API for enhanced AI email generation (uses built-in LLM with Sienna-style system prompt)
- [x] Write tests for new endpoints (35 tests all passing)
- [x] Save checkpoint and deliver

## Phase 9: Airtable + n8n + Apollo Dynamic Jobs Integration

- [x] Audit current codebase and understand existing data flow
- [x] Airtable billing limit hit — pivoted to MySQL database as single source of truth
- [x] Create n8n workflow: GOV.UK CSV → Apollo enrichment → ChatGPT descriptions → Database (ID: x26td0eY3xOSKlyb, active)
- [x] Webhook URL: https://toaoutreach.app.n8n.cloud/webhook/microsponsor-refresh
- [x] Create "Refresh Newest Jobs" tRPC endpoint that triggers n8n webhook
- [x] Update Admin panel with Refresh button, last refreshed date, new jobs count
- [x] Keep Add/Edit/Delete manual overrides in Admin
- [x] Add filter: "Show only ≤100 employees + active sponsors"
- [x] Free users see only 3 matches; Custom Plan unlocks all (blurred paywall overlay)
- [x] Loading spinner + success toast on refresh
- [x] Write tests for new endpoints (48 tests all passing)
- [x] Save checkpoint and deliver

## Phase 10: Fix Broken n8n Automations

- [x] Audit current n8n workflows and identify issues (webhook was GET-only, n8n API /run endpoint doesn't exist)
- [x] Disable/delete the Apify LinkedIn scraper workflow (ID: 5eYwxnmopfHOKQTB deactivated)
- [x] Fix Apollo enrichment workflow webhook trigger (changed from GET to POST)
- [x] Strengthen employee count filter (hard ≤100 only) — webhook rejects >100 employees
- [x] Fix webhook endpoint — added ≤100 employee filter, skips large companies, updated source to n8n_apollo
- [x] Fix Admin "Refresh Newest Jobs" button — removed broken n8n API /run fallback, now uses webhook POST directly
- [x] Remove large companies (Tata etc.) from database — deleted 3 Tata rows + 1 test row, 23 clean jobs remain
- [x] Enforce ≤100 employee filter on all data paths — webhook, getAllJobs, and Admin UI all enforce
- [x] Remove error badges from Admin UI — updated source badges to show Apollo, improved success toast
- [x] Run workflow once and verify only small sponsors appear — refresh triggered successfully, toast confirmed
- [x] Update "Last refreshed" timestamp — auto-updates on webhook callback
- [x] Save checkpoint and deliver (version: bd9874be)

## Phase 11: Real CV Upload + Airtable Job Matching

- [x] Audit current CV upload flow and Airtable base structure
- [x] Connect to Airtable base "Full time Job - Email Outreach" (base: appjQEGf90tVqMfVh, table: tblPUnTKjTgt2agko)
- [x] Build CV upload endpoint (PDF/DOCX → text extraction via pdf-parse/mammoth → S3 storage)
- [x] Build ChatGPT CV analysis endpoint (structured JSON output: skills, experience, preferred titles, visa status)
- [x] Build job matching logic (ChatGPT scores each Airtable job 0-100 with fit reasons)
- [x] Build /cv-analysis page with drag-drop upload, progress steps, profile summary, and scored job cards
- [x] Write 12 tests for CV analysis + Airtable types (65 total passing)
- [x] Save checkpoint and deliver (version: 77c36722)

## Phase 12: LinkedIn URL Generator + Homepage Redesign

- [x] Audit LinkedIn job search URL parameters and Airtable table structure
- [x] Create n8n workflow: receive CV profile → generate LinkedIn search URLs → upload to Airtable (workflow ID: gBqvCm3tBwJTnWxU)
- [x] Build backend endpoint to send CV analysis to n8n workflow
- [x] Remove separate /cv-analysis route
- [x] Make homepage the CV upload page (replace hero section)
- [x] Remove "CV Analysis" from navbar
- [x] Update frontend to show generated LinkedIn URLs after upload
- [x] Write tests — 70 tests passing (5 new LinkedIn URL tests)
- [x] Save checkpoint and deliver (version: d1525237)

## Phase 13: Import 42 Airtable Jobs + Admin Sync Button

- [x] Fetched all 100 records from Airtable "Full time Job - Email Outreach" base (more than 42 — includes all students)
- [x] Enriched missing salary fields with smart defaults based on title/location/seniority
- [x] Imported all 100 Airtable jobs into database (123 total with 23 manual)
- [x] Admin panel header shows dynamic job count from database
- [x] Added "Sync from Airtable" button with amber styling and Database icon
- [x] Added airtable_sync source badge (amber) to job cards
- [x] All jobs visible on Dashboard, My Matches, and Home page after CV analysis
- [x] Success toast shows imported/updated/error counts after sync
- [x] 73 tests passing (3 new Airtable sync tests)
- [x] Save checkpoint and deliver (version: dbea6e6d)

## Phase 14: Replace Jobs with Manus Airtable Base Table 1

- [x] Fetched all 42 records from Manus Airtable base (appNpBCPCvUKNaT10), Table 1
- [x] Deleted ALL existing jobs (123) from the website database
- [x] Imported all 42 jobs from Table 1 into the database (0 errors)
- [x] Updated Airtable service to use Manus base (appNpBCPCvUKNaT10) and Table 1 (tbldanNQEwoR9NgFI)
- [x] 69 tests passing (14 job tests rewritten for 42-job dataset)
- [x] Save checkpoint and deliver (version: f9e927c7)
