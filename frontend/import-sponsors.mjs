/**
 * Import GOV.UK Licensed Sponsors CSV into the database.
 * Only imports Skilled Worker route entries.
 * Uses batch inserts for performance.
 */
import mysql from "mysql2/promise";
import { createReadStream } from "fs";
import { createInterface } from "readline";

const connection = await mysql.createConnection(process.env.DATABASE_URL);

function parseCSVLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

async function importSponsors() {
  const filePath = "/home/ubuntu/upload/2026-03-13_-_Worker_and_Temporary_Worker(2).csv";
  
  const rl = createInterface({
    input: createReadStream(filePath, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });

  let lineNum = 0;
  let batch = [];
  const BATCH_SIZE = 500;
  let totalInserted = 0;
  let skipped = 0;

  for await (const line of rl) {
    lineNum++;
    if (lineNum === 1) continue; // Skip header

    const fields = parseCSVLine(line);
    if (fields.length < 5) {
      skipped++;
      continue;
    }

    const [orgName, townCity, county, typeRating, route] = fields;
    
    // Only import Skilled Worker route
    if (!route || !route.toLowerCase().includes("skilled worker")) {
      skipped++;
      continue;
    }

    if (!orgName || orgName.trim() === "") {
      skipped++;
      continue;
    }

    batch.push({
      organisationName: orgName.trim(),
      townCity: townCity?.trim() || null,
      county: county?.trim() || null,
      typeRating: typeRating?.trim() || null,
      route: route?.trim() || null,
      nameNormalised: orgName.trim().toLowerCase(),
    });

    if (batch.length >= BATCH_SIZE) {
      await connection.execute(buildInsertSQL(batch));
      totalInserted += batch.length;
      if (totalInserted % 5000 === 0) {
        console.log(`  Inserted ${totalInserted} sponsors...`);
      }
      batch = [];
    }
  }

  // Insert remaining
  if (batch.length > 0) {
    await connection.execute(buildInsertSQL(batch));
    totalInserted += batch.length;
  }

  console.log(`\nDone! Inserted ${totalInserted} Skilled Worker sponsors. Skipped ${skipped} non-matching rows.`);
  process.exit(0);
}

function escapeSQL(str) {
  if (str === null || str === undefined) return "NULL";
  return "'" + str.replace(/'/g, "''").replace(/\\/g, "\\\\") + "'";
}

function buildInsertSQL(rows) {
  const values = rows.map(r => {
    return `(${escapeSQL(r.organisationName)}, ${escapeSQL(r.townCity)}, ${escapeSQL(r.county)}, ${escapeSQL(r.typeRating)}, ${escapeSQL(r.route)}, ${escapeSQL(r.nameNormalised)}, NOW())`;
  }).join(",\n");
  
  return `INSERT INTO sponsors (organisationName, townCity, county, typeRating, route, nameNormalised, createdAt) VALUES ${values}`;
}

console.log("Starting GOV.UK Sponsor CSV import (Skilled Worker route only)...");
importSponsors().catch(err => {
  console.error("Import failed:", err);
  process.exit(1);
});
