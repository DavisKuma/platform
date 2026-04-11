/**
 * Import all 42 jobs from Manus Airtable base Table 1 into the website database.
 * Run: node import-manus-jobs.mjs
 */
import 'dotenv/config';
import { drizzle } from 'drizzle-orm/mysql2';
import mysql from 'mysql2/promise';
import { jobs } from './drizzle/schema.ts';
import fs from 'fs';

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) throw new Error('DATABASE_URL not set');

const pool = mysql.createPool(DATABASE_URL);
const db = drizzle(pool);

// Read the Airtable data
const raw = fs.readFileSync('/home/ubuntu/.mcp/tool-results/2026-03-15_20-05-28_airtable_list_records.json', 'utf-8');
const records = JSON.parse(raw);

function parseSalary(salaryStr) {
  if (!salaryStr || salaryStr === 'N/A' || salaryStr.trim() === '-' || salaryStr.trim() === '') {
    return { min: 0, max: 0 };
  }
  
  // Extract all numbers from the salary string
  const numbers = salaryStr.replace(/[,$]/g, '').match(/[\d.]+/g);
  if (!numbers || numbers.length === 0) return { min: 0, max: 0 };
  
  const vals = numbers.map(n => parseFloat(n));
  
  // Detect hourly/weekly and annualize
  const lower = salaryStr.toLowerCase();
  let multiplier = 1;
  if (lower.includes('hour')) multiplier = 2080; // 40h * 52w
  else if (lower.includes('week')) multiplier = 52;
  else if (lower.includes('month')) multiplier = 12;
  
  const min = Math.round(vals[0] * multiplier);
  const max = vals.length > 1 ? Math.round(vals[1] * multiplier) : min;
  
  return { min, max };
}

function guessIndustry(title, company, description) {
  const text = `${title} ${company} ${description}`.toLowerCase();
  if (text.includes('marketing') || text.includes('ecommerce') || text.includes('e-commerce')) return 'Marketing';
  if (text.includes('design') || text.includes('ux') || text.includes('ui')) return 'Design';
  if (text.includes('software') || text.includes('engineer') || text.includes('developer') || text.includes('web')) return 'Technology';
  if (text.includes('data') || text.includes('machine learning') || text.includes('ai') || text.includes('artificial intelligence')) return 'Technology';
  if (text.includes('finance') || text.includes('financial') || text.includes('analyst') || text.includes('investment') || text.includes('wealth')) return 'Finance';
  if (text.includes('healthcare') || text.includes('medical') || text.includes('orthodont') || text.includes('health')) return 'Healthcare';
  if (text.includes('real estate')) return 'Real Estate';
  if (text.includes('food') || text.includes('restaurant') || text.includes('pizza') || text.includes('domino')) return 'Food & Beverage';
  if (text.includes('sales') || text.includes('business development')) return 'Sales';
  if (text.includes('cloud') || text.includes('aws') || text.includes('amazon')) return 'Technology';
  if (text.includes('fashion') || text.includes('retail')) return 'Retail';
  return 'General';
}

async function main() {
  let imported = 0;
  let errors = 0;

  for (const rec of records) {
    const f = rec.fields || {};
    
    const company = f['Company'] || f['Name']?.split('-').pop()?.trim() || 'Unknown';
    const jobTitle = f['Job Titles'] || f['Name']?.split('-')[0]?.trim() || 'Unknown';
    const location = f['Location'] || 'United States';
    const salary = parseSalary(f['Salary']);
    const description = f['Job Description'] || '';
    const jobLink = f['Job Links'] || '';
    const companyUrl = f['Company URL'] || '';
    const hiringManager = f['Hiring manager'] || '';
    const hiringManagerEmail = f['Email (from Hiring manager)'] || '';
    const student = f['Student'] || '';
    const aiScore = f['AI Score'] || 0;
    const employeeCount = f['Employee Count'] || 0;
    const industry = guessIndustry(jobTitle, company, description);
    const emailDraft = f['Email Draft'] || '';
    const emailSubject = f['Email Subject'] || '';
    const approvalStatus = f['Approval Status'] || '';
    const status = f['Status'] || '';
    
    try {
      await db.insert(jobs).values({
        company: company.substring(0, 255),
        location: location.substring(0, 255),
        employeeCount: employeeCount || 0,
        jobTitle: jobTitle.substring(0, 255),
        salaryMin: salary.min,
        salaryMax: salary.max,
        isNewEntrantFriendly: true,
        isVerifiedSponsor: !!approvalStatus || status === 'Approved',
        hiringManager: hiringManager.substring(0, 255),
        hiringManagerEmail: hiringManagerEmail.substring(0, 320),
        companyDescription: description.substring(0, 10000) || `${company} - ${jobTitle}`,
        industry: industry.substring(0, 128),
        postedDays: 0,
        source: 'airtable_sync',
        linkedinUrl: jobLink || null,
        companyLinkedinUrl: companyUrl || null,
        linkedinDescription: description || null,
      });
      imported++;
      console.log(`  ✅ ${imported}. ${jobTitle} @ ${company} | ${location} | ${student}`);
    } catch (err) {
      errors++;
      console.error(`  ❌ Error: ${jobTitle} @ ${company}: ${err.message}`);
    }
  }

  console.log(`\n=== Import Complete ===`);
  console.log(`  Imported: ${imported}`);
  console.log(`  Errors: ${errors}`);
  console.log(`  Total records: ${records.length}`);

  await pool.end();
}

main().catch(console.error);
