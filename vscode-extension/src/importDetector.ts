/**
 * Detect MLToolKit imports without executing Python.
 * Strings and comments are masked before the import scan.
 */

export interface ImportScan {
  active: boolean;
  qualifiers: string[];
}

const PACKAGE_NAME = "MLToolKit";
const STRING_PREFIXES = new Set(["", "r", "u", "b", "f", "fr", "rf", "br", "rb", "bf", "fb", "fu", "uf"]);

export function maskNonCode(source: string): string {
  let output = "";
  let index = 0;
  while (index < source.length) {
    if (source[index] === "#") {
      const lineEnd = source.indexOf("\n", index);
      const end = lineEnd === -1 ? source.length : lineEnd;
      output += " ".repeat(end - index);
      index = end;
      continue;
    }
    const literal = matchStringStart(source, index);
    if (literal) {
      const end = scanString(source, literal.contentStart, literal.quote, literal.raw);
      output += " ".repeat(end - index);
      index = end;
      continue;
    }
    output += source[index];
    index += 1;
  }
  return output;
}

export function scanImports(source: string): ImportScan {
  const code = maskNonCode(source);
  const qualifiers = new Set<string>();
  let active = false;
  for (const statement of logicalStatements(code)) {
    const trimmed = statement.trim();
    if (/^import\s+/.test(trimmed)) {
      const clause = trimmed.replace(/^import\s+/, "");
      for (const part of splitTopLevel(clause, ",")) {
        const bound = bindingForImport(part.trim());
        if (bound) {
          active = true;
          qualifiers.add(bound);
        }
      }
      continue;
    }
    if (/^from\s+MLToolKit(?:\.[\w.]+)?\s+import\b/.test(trimmed)) {
      active = true;
    }
  }
  return { active, qualifiers: [...qualifiers].sort() };
}

export function findTypedCompletion(
  line: string,
  cursor: number,
  qualifiers: readonly string[]
): { qualifier: string; typed: string; startIndex: number } | null {
  const text = line.slice(0, cursor);
  const match = text.match(/(?:^|[^\w])([A-Za-z_]\w*)\.(ml_\w*)$/);
  if (!match || match.index === undefined) {
    return null;
  }
  const qualifier = match[1];
  if (!qualifiers.includes(qualifier)) {
    return null;
  }
  const typed = match[2];
  const full = `${qualifier}.${typed}`;
  return { qualifier, typed, startIndex: cursor - full.length };
}

function bindingForImport(part: string): string | null {
  const aliased = part.match(/^([\w.]+)\s+as\s+([A-Za-z_]\w*)$/);
  const moduleName = aliased ? aliased[1] : part;
  const alias = aliased ? aliased[2] : null;
  if (moduleName !== PACKAGE_NAME && !moduleName.startsWith(`${PACKAGE_NAME}.`)) {
    return null;
  }
  if (!/^[\w.]+$/.test(moduleName)) {
    return null;
  }
  return alias ?? moduleName.split(".")[0];
}

function logicalStatements(code: string): string[] {
  const statements: string[] = [];
  let current = "";
  let depth = 0;
  for (let index = 0; index < code.length; index += 1) {
    const character = code[index];
    if (character === "\\") {
      const next = code[index + 1];
      if (next === "\n") {
        current += " ";
        index += 1;
        continue;
      }
    }
    if (character === "(" || character === "[" || character === "{") {
      depth += 1;
    } else if (character === ")" || character === "]" || character === "}") {
      depth = Math.max(0, depth - 1);
    }
    if (character === "\n" && depth === 0) {
      pushStatements(statements, current);
      current = "";
      continue;
    }
    if (character === ";" && depth === 0) {
      pushStatements(statements, current);
      current = "";
      continue;
    }
    current += character === "\n" ? " " : character;
  }
  pushStatements(statements, current);
  return statements;
}

function pushStatements(statements: string[], chunk: string): void {
  const trimmed = chunk.trim();
  if (trimmed) {
    statements.push(trimmed);
  }
}

function splitTopLevel(value: string, separator: string): string[] {
  const parts: string[] = [];
  let current = "";
  let depth = 0;
  for (const character of value) {
    if (character === "(" || character === "[" || character === "{") {
      depth += 1;
    } else if (character === ")" || character === "]" || character === "}") {
      depth = Math.max(0, depth - 1);
    }
    if (character === separator && depth === 0) {
      parts.push(current);
      current = "";
      continue;
    }
    current += character;
  }
  parts.push(current);
  return parts;
}

function matchStringStart(
  source: string,
  index: number
): { contentStart: number; quote: string; raw: boolean } | null {
  const previous = index === 0 ? "" : source[index - 1];
  if (previous && /[A-Za-z0-9_]/.test(previous)) {
    return null;
  }
  let cursor = index;
  let prefix = "";
  while (cursor < source.length && prefix.length < 2 && /[rRuUbBfF]/.test(source[cursor])) {
    prefix += source[cursor];
    cursor += 1;
  }
  const normalized = prefix.toLowerCase().split("").sort().join("");
  if (!STRING_PREFIXES.has(normalized)) {
    return null;
  }
  const quoteChar = source[cursor];
  if (quoteChar !== "'" && quoteChar !== '"') {
    return null;
  }
  const triple = quoteChar.repeat(3);
  const quote = source.startsWith(triple, cursor) ? triple : quoteChar;
  return {
    contentStart: cursor + quote.length,
    quote,
    raw: normalized.includes("r"),
  };
}

function scanString(source: string, contentStart: number, quote: string, raw: boolean): number {
  let index = contentStart;
  while (index < source.length) {
    if (!raw && source[index] === "\\") {
      index += 2;
      continue;
    }
    if (source.startsWith(quote, index)) {
      return index + quote.length;
    }
    index += 1;
  }
  return source.length;
}
