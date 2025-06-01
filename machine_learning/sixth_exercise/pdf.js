import puppeteer from "puppeteer";

const browser = await puppeteer.launch(); // headless: true por padrão
const page = await browser.newPage();
await page.goto(`file://${process.cwd()}/Aula6_rennanguimaraes.html`, {
  waitUntil: "networkidle0",
});
await page.pdf({
  path: "Aula6_rennanguimaraes.pdf",
  format: "A4",
  printBackground: true, // mantém cores do tema dark
});
await browser.close();
console.log("PDF gerado com sucesso!");
