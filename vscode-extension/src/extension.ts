import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { findTypedCompletion, scanImports } from "./importDetector";
import { Catalog, filterSnippets, parseCatalog } from "./snippetEngine";

let catalog: Catalog = { package: "MLToolKit", version: "0", snippets: [] };

export function activate(context: vscode.ExtensionContext): void {
  try {
    catalog = parseCatalog(fs.readFileSync(path.join(context.extensionPath, "catalog.json"), "utf8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    void vscode.window.showErrorMessage(`MLToolKit could not read its snippet catalog: ${message}`);
  }

  const provider: vscode.CompletionItemProvider = {
    provideCompletionItems(document, position) {
      const config = vscode.workspace.getConfiguration("mltoolkit");
      if (!config.get("enable", true) || catalog.snippets.length === 0) {
        return undefined;
      }
      const scan = scanImports(activationSource(document));
      if (!scan.active || scan.qualifiers.length === 0) {
        return undefined;
      }
      const token = findTypedCompletion(document.lineAt(position).text, position.character, scan.qualifiers);
      if (!token) {
        return undefined;
      }
      const matches = filterSnippets(catalog.snippets, token.typed, config.get("maxItems", 300));
      if (matches.length === 0) {
        return undefined;
      }
      const range = new vscode.Range(new vscode.Position(position.line, token.startIndex), position);
      const typedToken = `${token.qualifier}.${token.typed}`;
      return matches.map((snippet) => {
        const item = new vscode.CompletionItem(
          `${token.qualifier}.${snippet.trigger}`,
          vscode.CompletionItemKind.Snippet
        );
        item.detail = snippet.category;
        item.documentation = new vscode.MarkdownString(documentationFor(snippet));
        item.insertText = new vscode.SnippetString(snippet.body);
        item.range = range;
        item.filterText = typedToken;
        item.sortText = `${snippet.category}:${snippet.trigger}`;
        return item;
      });
    },
  };

  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(
      [{ language: "python" }, { language: "jupyter" }],
      provider,
      ".",
      "_"
    )
  );
}

export function deactivate(): void {}

function activationSource(document: vscode.TextDocument): string {
  if (document.uri.scheme !== "vscode-notebook-cell") {
    return document.getText();
  }
  const notebook = vscode.workspace.notebookDocuments.find((candidate) =>
    candidate.getCells().some((cell) => cell.document.uri.toString() === document.uri.toString())
  );
  if (!notebook) {
    return document.getText();
  }
  return notebook
    .getCells()
    .filter((cell) => cell.document.languageId === "python")
    .map((cell) => cell.document.getText())
    .join("\n");
}

function documentationFor(snippet: Catalog["snippets"][number]): string {
  const variants = snippet.variants.slice(0, 8).map((name) => `\`${name}\``).join(", ");
  const variantLine = variants ? `\n\nRelated: ${variants}` : "";
  return `**${snippet.category}**\n\n${snippet.description}${variantLine}`;
}
