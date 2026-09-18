#!/usr/bin/env python3
"""
Markdown to LaTeX converter module.
Converts Markdown reports into LaTeX format while preserving the exact wording and structure.
"""

import re
import os


def escape_latex_text(text: str) -> str:
    """Escape LaTeX special characters outside math blocks and commands."""
    # Split by inline math: $...$
    parts = re.split(r'(\$[^\$]+\$)', text)
    escaped_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside inline math
            escaped_parts.append(part)
        else:
            # Outside math
            p = part
            p = re.sub(r'(?<!\\)%', r'\%', p)
            p = re.sub(r'(?<!\\)&', r'\&', p)
            p = re.sub(r'(?<!\\)#', r'\#', p)
            p = re.sub(r'(?<!\\)_', r'\_', p)
            escaped_parts.append(p)
    return ''.join(escaped_parts)


def convert_inline_styles(text: str) -> str:
    """Convert code, bold, and italic inline markdown styles."""
    # `code` -> \texttt{code}
    def code_repl(m):
        c = m.group(1)
        c = c.replace('\\', r'\textbackslash{}')
        c = c.replace('_', r'\_')
        c = c.replace('%', r'\%')
        c = c.replace('&', r'\&')
        c = c.replace('#', r'\#')
        return r'\texttt{' + c + '}'

    text = re.sub(r'`([^`]+)`', code_repl, text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\textit{\1}', text)
    return text


def convert_table(table_lines: list) -> str:
    """Convert a markdown table to a LaTeX tabular."""
    rows = []
    for line in table_lines:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        rows.append(cells)

    if len(rows) < 2:
        return '\n'.join(table_lines) + '\n'

    is_delim = all(re.match(r'^:?-+:?$', c) for c in rows[1])
    if is_delim:
        header = rows[0]
        data_rows = rows[2:]
    else:
        header = None
        data_rows = rows

    num_cols = max(len(r) for r in rows)
    col_spec = 'l' + 'c' * (num_cols - 1) if num_cols > 1 else 'l'

    tex = []
    tex.append(r'\begin{center}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{' + col_spec + '}')
    tex.append(r'\toprule')

    if header:
        formatted_header = [convert_inline_styles(escape_latex_text(c)) for c in header]
        tex.append(' & '.join([r'\textbf{' + c + '}' if c else '' for c in formatted_header]) + r' \\')
        tex.append(r'\midrule')

    for r in data_rows:
        while len(r) < num_cols:
            r.append('')
        formatted_row = [convert_inline_styles(escape_latex_text(c)) for c in r]
        tex.append(' & '.join(formatted_row) + r' \\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{center}')
    return '\n'.join(tex) + '\n'


def markdown_to_tex(md_text: str, base_heading_level: int = 1) -> str:
    """
    Convert a Markdown string into LaTeX syntax.
    base_heading_level: 1 means # -> \section, 2 means # -> \subsection, etc.
    """
    lines = md_text.splitlines()
    out = []

    in_code_block = False
    code_block_lines = []

    in_math_block = False
    math_block_lines = []

    in_table = False
    table_lines = []

    list_stack = []

    def close_lists():
        nonlocal list_stack, out
        while list_stack:
            env = list_stack.pop()
            out.append(f'\\end{{{env}}}')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            if in_code_block:
                out.append(r'\begin{verbatim}')
                out.extend(code_block_lines)
                out.append(r'\end{verbatim}')
                in_code_block = False
                code_block_lines = []
            else:
                close_lists()
                in_code_block = True
                code_block_lines = []
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Math block $$
        if line.strip() == '$$':
            if in_math_block:
                out.append(r'\]')
                in_math_block = False
            else:
                close_lists()
                in_math_block = True
                out.append(r'\[')
            i += 1
            continue

        if line.strip().startswith('$$') and line.strip().endswith('$$') and len(line.strip()) > 4:
            close_lists()
            math_content = line.strip()[2:-2].strip()
            out.append(r'\[' + math_content + r'\]')
            i += 1
            continue

        if in_math_block:
            out.append(line)
            i += 1
            continue

        # Table
        if line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                close_lists()
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                out.append(convert_table(table_lines))
                in_table = False
                table_lines = []

        # Horizontal rule
        if re.match(r'^---+$', line.strip()):
            close_lists()
            out.append(r'\vspace{0.5em}\hrule\vspace{0.5em}')
            i += 1
            continue

        # Headings
        m_h = re.match(r'^(#+)\s*(.*)$', line)
        if m_h:
            close_lists()
            level = len(m_h.group(1)) + (base_heading_level - 1)
            title = m_h.group(2).strip()
            title_tex = convert_inline_styles(escape_latex_text(title))
            if level == 1:
                out.append(f'\\section{{{title_tex}}}')
            elif level == 2:
                out.append(f'\\subsection{{{title_tex}}}')
            elif level == 3:
                out.append(f'\\subsubsection{{{title_tex}}}')
            elif level == 4:
                out.append(f'\\paragraph{{{title_tex}}}')
            else:
                out.append(f'\\subparagraph{{{title_tex}}}')
            i += 1
            continue

        # Lists
        m_item = re.match(r'^(\s*)[*-]\s+(.*)$', line)
        m_enum = re.match(r'^(\s*)\d+\.\s+(.*)$', line)

        if m_item:
            item_text = m_item.group(2).strip()
            if not list_stack or list_stack[-1] != 'itemize':
                if list_stack and list_stack[-1] == 'enumerate':
                    out.append(r'\end{enumerate}')
                    list_stack.pop()
                out.append(r'\begin{itemize}')
                list_stack.append('itemize')
            formatted_item = convert_inline_styles(escape_latex_text(item_text))
            out.append(f'\\item {formatted_item}')
            i += 1
            continue
        elif m_enum:
            item_text = m_enum.group(2).strip()
            if not list_stack or list_stack[-1] != 'enumerate':
                if list_stack and list_stack[-1] == 'itemize':
                    out.append(r'\end{itemize}')
                    list_stack.pop()
                out.append(r'\begin{enumerate}')
                list_stack.append('enumerate')
            formatted_item = convert_inline_styles(escape_latex_text(item_text))
            out.append(f'\\item {formatted_item}')
            i += 1
            continue
        else:
            if line.strip() == '':
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if not re.match(r'^\s*([*-]|\d+\.)\s+', next_line):
                        close_lists()
                else:
                    close_lists()
                out.append('')
                i += 1
                continue

        # Regular paragraph text
        close_lists()
        formatted_line = convert_inline_styles(escape_latex_text(line))
        out.append(formatted_line)
        i += 1

    close_lists()
    if in_table:
        out.append(convert_table(table_lines))

    return '\n'.join(out)


def convert_md_file_to_tex(md_path: str, tex_path: str, base_heading_level: int = 1):
    """Read a markdown file and write its LaTeX counterpart."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tex_content = markdown_to_tex(content, base_heading_level=base_heading_level)
    os.makedirs(os.path.dirname(os.path.abspath(tex_path)), exist_ok=True)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"Generated TeX report at: {tex_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        convert_md_file_to_tex(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        in_p = sys.argv[1]
        out_p = os.path.splitext(in_p)[0] + '.tex'
        convert_md_file_to_tex(in_p, out_p)
    else:
        print("Usage: python md_to_tex.py <input.md> [output.tex]")
