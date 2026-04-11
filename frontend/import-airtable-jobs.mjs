/**
 * Import all Airtable jobs into the website database.
 * Uses the existing upsertJobFromWebhook function.
 * Run with: node import-airtable-jobs.mjs
 */
import { readFileSync } from "fs";
import { drizzle } from "drizzle-orm/mysql2";
import { eq, and, desc } from "drizzle-orm";
import { int, mysqlTable, varchar, text, boolean, timestamp, mysqlEnum } from "drizzle-orm/mysql-core";

// Re-define the jobs table schema inline (to avoid TS import issues)
const jobs = mysqlTable("jobs", {
  id: int("id").autoincrement().primaryKey(),
  company: varchar("company", { length: 255 }).notNull(),
  location: varchar("location", { length: 255 }).notNull(),
  employeeCount: int("employeeCount").notNull(),
  jobTitle: varchar("jobTitle", { length: 255 }).notNull(),
  salaryMin: int("salaryMin").notNull(),
  salaryMax: int("salaryMax").notNull(),
  isNewEntrantFriendly: boolean("isNewEntrantFriendly").default(true).notNull(),
  isVerifiedSponsor: boolean("isVerifiedSponsor").default(true).notNull(),
  hiringManager: varchar("hiringManager", { length: 255 }).notNull(),
  hiringManagerEmail: varchar("hiringManagerEmail", { length: 320 }).notNull(),
  companyDescription: text("companyDescription").notNull(),
  industry: varchar("industry", { length: 128 }).notNull(),
  postedDays: int("postedDays").default(0).notNull(),
  source: varchar("source", { length: 64 }).default("manual"),
  linkedinUrl: text("linkedinUrl"),
  companyLinkedinUrl: text("companyLinkedinUrl"),
  linkedinDescription: text("linkedinDescription"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

const emailTemplates = mysqlTable("email_templates", {
  id: int("id").autoincrement().primaryKey(),
  jobId: int("jobId").notNull(),
  subject: varchar("subject", { length: 500 }).notNull(),
  body: text("body").notNull(),
  company: varchar("company", { length: 255 }).notNull(),
  hiringManager: varchar("hiringManager", { length: 255 }).notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

// Read the parsed Airtable jobs
const airtableJobs = JSON.parse(readFileSync("/home/ubuntu/airtable-jobs.json", "utf-8"));

// Guess industry from job title and description
function guessIndustry(title, desc) {
  const text = `${title} ${desc}`.toLowerCase();
  if (text.includes("marketing") || text.includes("seo") || text.includes("social media") || text.includes("content")) return "Marketing";
  if (text.includes("software") || text.includes("developer") || text.includes("engineer") || text.includes("data") || text.includes("tech") || text.includes("ai") || text.includes("machine learning")) return "Technology";
  if (text.includes("finance") || text.includes("accounting") || text.includes("financial") || text.includes("investment") || text.includes("banking")) return "Finance";
  if (text.includes("sales") || text.includes("sdr") || text.includes("business development") || text.includes("account executive")) return "Sales";
  if (text.includes("design") || text.includes("ux") || text.includes("ui") || text.includes("creative") || text.includes("graphic")) return "Design";
  if (text.includes("operations") || text.includes("logistics") || text.includes("supply chain") || text.includes("vendor")) return "Operations";
  if (text.includes("real estate") || text.includes("property")) return "Real Estate";
  if (text.includes("legal") || text.includes("compliance") || text.includes("regulatory")) return "Legal";
  if (text.includes("hr") || text.includes("human resources") || text.includes("recruiting") || text.includes("talent")) return "Human Resources";
  if (text.includes("consulting") || text.includes("analyst") || text.includes("strategy")) return "Consulting";
  if (text.includes("healthcare") || text.includes("medical") || text.includes("health")) return "Healthcare";
  if (text.includes("education") || text.includes("teaching") || text.includes("training")) return "Education";
  if (text.includes("media") || text.includes("journalism") || text.includes("publishing")) return "Media";
  return "General";
}

// Guess salary range from location and job title
function guessSalary(title, location) {
  const loc = location.toLowerCase();
  const t = title.toLowerCase();
  const isUS = loc.includes("united states") || loc.includes("new york") || loc.includes("san francisco") || loc.includes("houston") || loc.includes(", ca") || loc.includes(", ny") || loc.includes(", tx") || loc.includes("bay area");
  const isUK = loc.includes("united kingdom") || loc.includes("london") || loc.includes("england") || loc.includes("manchester") || loc.includes("birmingham");
  
  // Senior/manager roles
  const isSenior = t.includes("senior") || t.includes("manager") || t.includes("lead") || t.includes("director") || t.includes("head");
  const isIntern = t.includes("intern") || t.includes("trainee") || t.includes("apprentice");
  
  if (isUS) {
    if (isIntern) return { min: 35000, max: 50000 };
    if (isSenior) return { min: 80000, max: 120000 };
    return { min: 50000, max: 75000 };
  }
  if (isUK) {
    if (isIntern) return { min: 22000, max: 28000 };
    if (isSenior) return { min: 45000, max: 70000 };
    return { min: 28000, max: 42000 };
  }
  // Default
  if (isIntern) return { min: 25000, max: 35000 };
  if (isSenior) return { min: 50000, max: 80000 };
  return { min: 30000, max: 50000 };
}

// Parse salary string if available
function parseSalary(salaryStr, title, location) {
  if (!salaryStr || salaryStr.trim() === "-" || salaryStr.trim() === " - ") {
    return guessSalary(title, location);
  }
  // Try to extract numbers
  const numbers = salaryStr.match(/[\d,]+/g);
  if (numbers && numbers.length >= 2) {
    const min = parseInt(numbers[0].replace(/,/g, ""));
    const max = parseInt(numbers[1].replace(/,/g, ""));
    if (min > 0 && max > 0) return { min, max };
  }
  if (numbers && numbers.length === 1) {
    const val = parseInt(numbers[0].replace(/,/g, ""));
    if (val > 0) return { min: val, max: Math.round(val * 1.2) };
  }
  return guessSalary(title, location);
}

// Calculate posted days from date
function calcPostedDays(dateStr) {
  if (!dateStr) return 7;
  try {
    const posted = new Date(dateStr);
    const now = new Date();
    const diff = Math.floor((now - posted) / (1000 * 60 * 60 * 24));
    return Math.max(0, diff);
  } catch {
    return 7;
  }
}

async function main() {
  const db = drizzle(process.env.DATABASE_URL);
  
  let imported = 0;
  let updated = 0;
  let errors = 0;

  for (const aj of airtableJobs) {
    try {
      const salary = parseSalary(aj.salary, aj.job_title, aj.location);
      const industry = guessIndustry(aj.job_title, aj.job_description);
      const postedDays = calcPostedDays(aj.job_posted_date);
      
      // Truncate description for companyDescription field
      const companyDesc = aj.job_description ? aj.job_description.substring(0, 500) : "";
      
      // Check if job already exists
      const existing = await db.select({ id: jobs.id })
        .from(jobs)
        .where(and(
          eq(jobs.company, aj.company),
          eq(jobs.jobTitle, aj.job_title)
        ))
        .limit(1);

      if (existing.length > 0) {
        // Update existing
        await db.update(jobs).set({
          location: aj.location || "United Kingdom",
          salaryMin: salary.min,
          salaryMax: salary.max,
          industry,
          postedDays,
          source: "airtable_sync",
          linkedinUrl: aj.job_link || null,
          companyLinkedinUrl: aj.company_url || null,
          linkedinDescription: aj.job_description || null,
          companyDescription: companyDesc,
        }).where(eq(jobs.id, existing[0].id));
        
        // Also upsert email template if we have one
        if (aj.email_draft && aj.email_subject) {
          const existingTemplate = await db.select({ id: emailTemplates.id })
            .from(emailTemplates)
            .where(eq(emailTemplates.jobId, existing[0].id))
            .limit(1);
          
          if (existingTemplate.length === 0) {
            await db.insert(emailTemplates).values({
              jobId: existing[0].id,
              subject: aj.email_subject,
              body: aj.email_draft,
              company: aj.company,
              hiringManager: "Hiring Manager",
            });
          }
        }
        
        updated++;
        console.log(`  Updated: ${aj.job_title} @ ${aj.company}`);
      } else {
        // Insert new
        await db.insert(jobs).values({
          company: aj.company,
          location: aj.location || "United Kingdom",
          employeeCount: 50, // Default for Airtable jobs
          jobTitle: aj.job_title,
          salaryMin: salary.min,
          salaryMax: salary.max,
          isNewEntrantFriendly: true,
          isVerifiedSponsor: true,
          hiringManager: "Hiring Manager",
          hiringManagerEmail: "careers@" + aj.company.toLowerCase().replace(/[^a-z0-9]/g, "") + ".com",
          companyDescription: companyDesc,
          industry,
          postedDays,
          source: "airtable_sync",
          linkedinUrl: aj.job_link || null,
          companyLinkedinUrl: aj.company_url || null,
          linkedinDescription: aj.job_description || null,
        });
        
        // Get the inserted job ID for email template
        if (aj.email_draft && aj.email_subject) {
          const [inserted] = await db.select({ id: jobs.id })
            .from(jobs)
            .where(and(eq(jobs.company, aj.company), eq(jobs.jobTitle, aj.job_title)))
            .orderBy(desc(jobs.createdAt))
            .limit(1);
          
          if (inserted) {
            await db.insert(emailTemplates).values({
              jobId: inserted.id,
              subject: aj.email_subject,
              body: aj.email_draft,
              company: aj.company,
              hiringManager: "Hiring Manager",
            });
          }
        }
        
        imported++;
        console.log(`  Imported: ${aj.job_title} @ ${aj.company}`);
      }
    } catch (err) {
      errors++;
      console.error(`  ERROR: ${aj.job_title} @ ${aj.company}: ${err.message}`);
    }
  }

  // Get final count
  const [countResult] = await db.select({ count: int("id") }).from(jobs);
  
  console.log(`\n=== Import Summary ===`);
  console.log(`New jobs imported: ${imported}`);
  console.log(`Existing jobs updated: ${updated}`);
  console.log(`Errors: ${errors}`);
  console.log(`Total Airtable records processed: ${airtableJobs.length}`);
  
  process.exit(0);
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});
