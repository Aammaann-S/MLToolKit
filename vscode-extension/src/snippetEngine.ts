export interface CatalogSnippet {
  trigger: string;
  displayName: string;
  description: string;
  category: string;
  categories: string[];
  aliases: string[];
  imports: string[];
  body: string;
  placeholders: { index: string; default: string }[];
  implementation: string;
  variants: string[];
}

export interface Catalog {
  package: string;
  version: string;
  snippets: CatalogSnippet[];
}

export function filterSnippets(snippets: CatalogSnippet[], typed: string, maxItems = 300): CatalogSnippet[] {
  const query = typed.toLowerCase();
  const matches = snippets
    .filter((snippet) => keysOf(snippet).some((key) => key.toLowerCase().startsWith(query)))
    .sort((left, right) => left.category.localeCompare(right.category) || left.trigger.localeCompare(right.trigger));
  if (maxItems > 0 && matches.length > maxItems) {
    return matches.slice(0, maxItems);
  }
  return matches;
}

export function keysOf(snippet: CatalogSnippet): string[] {
  return [snippet.trigger, ...(snippet.aliases ?? [])];
}

export function parseCatalog(raw: string): Catalog {
  const parsed = JSON.parse(raw.replace(/^\uFEFF/, "")) as Catalog;
  if (!parsed || !Array.isArray(parsed.snippets)) {
    throw new Error("MLToolKit catalog is missing its snippet list");
  }
  return parsed;
}
