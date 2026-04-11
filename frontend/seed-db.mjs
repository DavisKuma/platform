import { drizzle } from "drizzle-orm/mysql2";
import { mysqlTable, int, varchar, text, boolean, timestamp } from "drizzle-orm/mysql-core";
import { sql } from "drizzle-orm";
import dotenv from "dotenv";

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
  console.error("DATABASE_URL not set");
  process.exit(1);
}

const db = drizzle(DATABASE_URL);

// Inline table definitions for the seed script (since we can't import .ts)
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
});

const emailTemplates = mysqlTable("email_templates", {
  id: int("id").autoincrement().primaryKey(),
  jobId: int("jobId").notNull(),
  subject: varchar("subject", { length: 500 }).notNull(),
  body: text("body").notNull(),
  company: varchar("company", { length: 255 }).notNull(),
  hiringManager: varchar("hiringManager", { length: 255 }).notNull(),
});

const demoCvs = mysqlTable("demo_cvs", {
  id: int("id").autoincrement().primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  university: varchar("university", { length: 255 }).notNull(),
  degree: varchar("degree", { length: 255 }).notNull(),
  skills: text("skills").notNull(),
  experience: text("experience").notNull(),
  visaStatus: varchar("visaStatus", { length: 255 }).notNull(),
  isDefault: boolean("isDefault").default(false).notNull(),
});

const jobListings = [
  { company: "SecondWind Pro", location: "London, UK", employeeCount: 34, jobTitle: "UX Designer", salaryMin: 35000, salaryMax: 45000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Luke Harrison", hiringManagerEmail: "luke.h@secondwindpro.com", companyDescription: "Sports analytics platform helping athletes track performance and market value through data-driven dashboards", industry: "Sports Tech", postedDays: 2 },
  { company: "Zoonou Limited", location: "Manchester, UK", employeeCount: 67, jobTitle: "QA & UX Analyst", salaryMin: 30000, salaryMax: 40000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Sarah Mitchell", hiringManagerEmail: "s.mitchell@zoonou.com", companyDescription: "Software testing consultancy specialising in accessibility and user experience validation for enterprise clients", industry: "Software Testing", postedDays: 5 },
  { company: "DataVista Solutions", location: "Bristol, UK", employeeCount: 22, jobTitle: "Product Designer", salaryMin: 38000, salaryMax: 48000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "James Okonkwo", hiringManagerEmail: "james@datavista.co.uk", companyDescription: "Data visualisation startup transforming complex datasets into intuitive, interactive dashboards for SMEs", industry: "Data Analytics", postedDays: 1 },
  { company: "Greenfield Digital", location: "Edinburgh, UK", employeeCount: 45, jobTitle: "Junior UX Researcher", salaryMin: 28000, salaryMax: 35000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Emma Blackwood", hiringManagerEmail: "emma.b@greenfielddigital.co.uk", companyDescription: "Sustainable tech agency building digital products for environmental organisations and green energy startups", industry: "GreenTech", postedDays: 3 },
  { company: "Nexus Health AI", location: "Cambridge, UK", employeeCount: 18, jobTitle: "Design Strategist", salaryMin: 40000, salaryMax: 52000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Dr. Priya Sharma", hiringManagerEmail: "priya@nexushealthai.com", companyDescription: "AI-powered health diagnostics platform using machine learning to improve early disease detection in NHS trusts", industry: "HealthTech", postedDays: 4 },
  { company: "Craft Studio UK", location: "Brighton, UK", employeeCount: 12, jobTitle: "UI/UX Designer", salaryMin: 32000, salaryMax: 42000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Tom Whitfield", hiringManagerEmail: "tom@craftstudio.co.uk", companyDescription: "Boutique design agency creating premium brand experiences for luxury and lifestyle clients across Europe", industry: "Design Agency", postedDays: 7 },
  { company: "FinLedger", location: "London, UK", employeeCount: 56, jobTitle: "Product Designer", salaryMin: 42000, salaryMax: 55000, isNewEntrantFriendly: false, isVerifiedSponsor: true, hiringManager: "Rachel Kim", hiringManagerEmail: "rachel.k@finledger.io", companyDescription: "Blockchain-based accounting platform simplifying financial compliance for UK startups and SMEs", industry: "FinTech", postedDays: 6 },
  { company: "LearnLoop", location: "Leeds, UK", employeeCount: 29, jobTitle: "UX/UI Designer", salaryMin: 30000, salaryMax: 38000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "David Chen", hiringManagerEmail: "david@learnloop.edu", companyDescription: "EdTech platform gamifying professional development for corporate teams with adaptive learning paths", industry: "EdTech", postedDays: 2 },
  { company: "Orbital Robotics", location: "Oxford, UK", employeeCount: 41, jobTitle: "Interaction Designer", salaryMin: 36000, salaryMax: 46000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Alex Petrov", hiringManagerEmail: "alex.p@orbitalrobotics.co.uk", companyDescription: "Robotics startup developing autonomous warehouse systems with intuitive human-robot interfaces", industry: "Robotics", postedDays: 8 },
  { company: "Mosaic Media", location: "Birmingham, UK", employeeCount: 15, jobTitle: "Creative Designer", salaryMin: 28000, salaryMax: 36000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Fatima Al-Rashid", hiringManagerEmail: "fatima@mosaicmedia.co.uk", companyDescription: "Multicultural media agency producing content and campaigns that celebrate diversity in British culture", industry: "Media & Marketing", postedDays: 3 },
  { company: "CloudNest", location: "Glasgow, UK", employeeCount: 73, jobTitle: "Frontend Designer", salaryMin: 34000, salaryMax: 44000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Ian McGregor", hiringManagerEmail: "ian@cloudnest.tech", companyDescription: "Cloud infrastructure company providing scalable hosting solutions tailored for UK government and NHS projects", industry: "Cloud Computing", postedDays: 5 },
  { company: "TrueNorth Legal", location: "London, UK", employeeCount: 38, jobTitle: "Service Designer", salaryMin: 37000, salaryMax: 47000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Charlotte Webb", hiringManagerEmail: "c.webb@truenorthlegal.com", companyDescription: "LegalTech startup digitising immigration and visa processes for UK law firms and their international clients", industry: "LegalTech", postedDays: 1 },
  { company: "PetPal UK", location: "Cardiff, UK", employeeCount: 8, jobTitle: "UX Designer", salaryMin: 26000, salaryMax: 34000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Megan Lloyd", hiringManagerEmail: "megan@petpal.co.uk", companyDescription: "Pet wellness app connecting pet owners with local vets, groomers, and pet-friendly venues across the UK", industry: "PetTech", postedDays: 4 },
  { company: "Aether Audio", location: "Sheffield, UK", employeeCount: 52, jobTitle: "Product Designer", salaryMin: 33000, salaryMax: 43000, isNewEntrantFriendly: false, isVerifiedSponsor: true, hiringManager: "Marcus Reed", hiringManagerEmail: "marcus@aetheraudio.com", companyDescription: "Audio technology company developing spatial sound solutions for immersive gaming and virtual reality experiences", industry: "Audio Tech", postedDays: 9 },
  { company: "Bloom Gardens", location: "Bath, UK", employeeCount: 19, jobTitle: "Digital Designer", salaryMin: 29000, salaryMax: 37000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Sophie Turner", hiringManagerEmail: "sophie@bloomgardens.co.uk", companyDescription: "D2C garden design platform using AR to help homeowners visualise and plan their outdoor spaces before purchase", industry: "PropTech", postedDays: 6 },
  { company: "Kinetic Labs", location: "Nottingham, UK", employeeCount: 31, jobTitle: "UX Researcher", salaryMin: 32000, salaryMax: 40000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Oliver Grant", hiringManagerEmail: "oliver@kineticlabs.io", companyDescription: "Wearable fitness technology startup creating smart clothing that tracks biomechanics for injury prevention", industry: "Wearable Tech", postedDays: 2 },
  { company: "Hive Collective", location: "Liverpool, UK", employeeCount: 24, jobTitle: "Brand & UX Designer", salaryMin: 30000, salaryMax: 39000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Zara Hussain", hiringManagerEmail: "zara@hivecollective.co.uk", companyDescription: "Social enterprise incubator helping underrepresented founders build and launch tech startups in the North", industry: "Social Enterprise", postedDays: 3 },
  { company: "Quantum Bridge", location: "London, UK", employeeCount: 88, jobTitle: "Senior UX Designer", salaryMin: 50000, salaryMax: 65000, isNewEntrantFriendly: false, isVerifiedSponsor: true, hiringManager: "Michael Torres", hiringManagerEmail: "m.torres@quantumbridge.tech", companyDescription: "Quantum computing consultancy making quantum algorithms accessible through intuitive developer tools and APIs", industry: "Quantum Computing", postedDays: 10 },
  { company: "FreshPlate", location: "Bristol, UK", employeeCount: 42, jobTitle: "UX/Product Designer", salaryMin: 34000, salaryMax: 44000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Lily Nguyen", hiringManagerEmail: "lily@freshplate.co.uk", companyDescription: "Sustainable food delivery platform connecting consumers with local farms for zero-waste, seasonal meal kits", industry: "FoodTech", postedDays: 1 },
  { company: "Atlas Compliance", location: "London, UK", employeeCount: 61, jobTitle: "Design Lead", salaryMin: 45000, salaryMax: 58000, isNewEntrantFriendly: false, isVerifiedSponsor: true, hiringManager: "Andrew Blake", hiringManagerEmail: "a.blake@atlascompliance.com", companyDescription: "RegTech platform automating regulatory compliance reporting for financial services firms across the UK and EU", industry: "RegTech", postedDays: 7 },
  { company: "Ripple Energy", location: "Newcastle, UK", employeeCount: 36, jobTitle: "Product Designer", salaryMin: 33000, salaryMax: 42000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Hannah Brooks", hiringManagerEmail: "hannah@rippleenergy.co.uk", companyDescription: "Community-owned renewable energy platform enabling households to invest in and benefit from local wind farms", industry: "CleanTech", postedDays: 4 },
  { company: "Pixel Forge", location: "Reading, UK", employeeCount: 27, jobTitle: "Game UX Designer", salaryMin: 35000, salaryMax: 45000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Jake Morrison", hiringManagerEmail: "jake@pixelforge.games", companyDescription: "Indie game studio creating narrative-driven mobile games with a focus on mental health and wellbeing themes", industry: "Gaming", postedDays: 5 },
  { company: "Beacon Insights", location: "London, UK", employeeCount: 48, jobTitle: "Data Visualisation Designer", salaryMin: 38000, salaryMax: 50000, isNewEntrantFriendly: true, isVerifiedSponsor: true, hiringManager: "Nadia Petrova", hiringManagerEmail: "nadia@beaconinsights.co.uk", companyDescription: "Business intelligence consultancy creating bespoke data storytelling solutions for FTSE 250 companies", industry: "Business Intelligence", postedDays: 2 },
];

const emailTemplatesData = [
  { jobId: 1, subject: "Luke, SecondWind Pro: UX help for athlete valuation dashboards?", company: "SecondWind Pro", hiringManager: "Luke Harrison", body: "Hi Luke,\n\nI came across SecondWind Pro's athlete performance dashboards and was genuinely impressed — the way you're turning raw performance data into actionable insights for athletes and agents is exactly the kind of product challenge I love.\n\nI noticed your valuation dashboard could benefit from some UX refinements, particularly around how athletes compare their metrics over time. I've sketched a few ideas on how progressive disclosure could simplify the data-heavy screens without losing depth.\n\nI'm Sienna Chen, an MS Strategic Design student at Parsons with 2 years of product design experience at a Shanghai fintech startup where I led the redesign of a mobile trading app serving 500K+ users. I specialise in making complex data feel intuitive — which seems like exactly what SecondWind Pro needs as you scale.\n\nI'm currently on a Graduate visa, and as a New Entrant, the reduced salary threshold makes sponsoring me genuinely low-risk for a company your size. I'd love to share my portfolio and those dashboard ideas over a quick 15-minute call.\n\nWould Thursday or Friday work for you?\n\nBest,\nSienna Chen\nPortfolio: sienna-chen.design\nLinkedIn: linkedin.com/in/siennachen" },
  { jobId: 3, subject: "James, DataVista: turning complex data into stories that click?", company: "DataVista Solutions", hiringManager: "James Okonkwo", body: "Hi James,\n\nI've been following DataVista's work on making enterprise data accessible to SMEs — your interactive dashboard demos on LinkedIn are a masterclass in data storytelling. The way you handle multi-dimensional filtering is particularly clever.\n\nI'm Sienna Chen, completing my MS in Strategic Design at Parsons. Before grad school, I spent 2 years at a Shanghai fintech startup redesigning a trading app for 500K+ users — essentially turning overwhelming financial data into clear, actionable interfaces. That experience maps directly to what DataVista is building.\n\nI have some specific ideas about how micro-interactions could make your dashboard onboarding smoother for first-time SME users who aren't data-literate. I'd love to walk you through them.\n\nI'm on a Graduate visa — as a New Entrant, the reduced salary threshold means sponsoring me is genuinely low-risk and cost-effective for a growing team like yours.\n\nCould we find 15 minutes this week for a quick chat?\n\nBest,\nSienna Chen\nPortfolio: sienna-chen.design\nLinkedIn: linkedin.com/in/siennachen" },
  { jobId: 5, subject: "Dr. Sharma, Nexus Health AI: designing trust into diagnostic interfaces?", company: "Nexus Health AI", hiringManager: "Dr. Priya Sharma", body: "Hi Dr. Sharma,\n\nNexus Health AI's work on early disease detection using ML is extraordinary — the potential impact on NHS trusts is massive. I read your recent case study on reducing diagnostic turnaround times by 40%, and it got me thinking about the design challenges of presenting AI-generated health insights to clinicians who need to trust the output instantly.\n\nI'm Sienna Chen, an MS Strategic Design student at Parsons with a background in making complex, high-stakes data feel trustworthy and intuitive. At my previous role in a Shanghai fintech startup, I redesigned interfaces where users made real-money decisions based on data visualisations — the trust design principles are remarkably similar to health diagnostics.\n\nI've been researching how progressive confidence indicators and contextual explanations can help clinicians calibrate their trust in AI recommendations. I'd love to share my thinking with your team.\n\nI'm on a Graduate visa, and as a New Entrant, the reduced salary threshold makes sponsoring me a low-risk investment. I'd welcome the chance to discuss how my design strategy skills could support Nexus Health AI's mission.\n\nWould you have 15 minutes this week?\n\nWarm regards,\nSienna Chen\nPortfolio: sienna-chen.design\nLinkedIn: linkedin.com/in/siennachen" },
];

const demoCvData = {
  name: "Sienna Chen",
  university: "Parsons School of Design",
  degree: "MS Strategic Design & Management",
  skills: JSON.stringify(["UX Research", "Product Strategy", "Design Thinking", "Figma", "User Testing", "Data Visualisation"]),
  experience: "2 years product design at a Shanghai fintech startup, led redesign of mobile trading app serving 500K+ users",
  visaStatus: "Graduate Visa (2-year post-study work)",
  isDefault: true,
};

async function seed() {
  console.log("Seeding database...");

  // Clear existing data
  await db.delete(emailTemplates);
  await db.delete(demoCvs);
  await db.delete(jobs);
  console.log("✓ Cleared existing data");

  // Insert jobs
  console.log("Inserting 23 job listings...");
  await db.insert(jobs).values(jobListings);
  console.log("✓ Jobs inserted");

  // Insert email templates
  console.log("Inserting email templates...");
  await db.insert(emailTemplates).values(emailTemplatesData);
  console.log("✓ Email templates inserted");

  // Insert demo CV
  console.log("Inserting demo CV...");
  await db.insert(demoCvs).values(demoCvData);
  console.log("✓ Demo CV inserted");

  console.log("\nDatabase seeded successfully!");
  process.exit(0);
}

seed().catch((err) => {
  console.error("Seed failed:", err);
  process.exit(1);
});
