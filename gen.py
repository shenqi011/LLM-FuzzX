import os

def not_include_file(filename):
    return filename == "gen.py" or filename == "_all_code.txt"

def is_code_file(filename):
    # 根据需要的文件后缀名来判断
    return (
        not not_include_file(filename) and (
            filename.endswith('.py') or
            filename.endswith('.html') or
            filename.endswith('.java') or
            filename.endswith('.c') or
            filename.endswith('.cpp') or
            filename.endswith('.js') or
            filename.endswith('.ts') or
            filename.endswith('.vue') or
            filename.endswith('.properties') or
            filename.endswith('.md') or
            filename == "tsconfig.json" or
            filename == "package.json" or
            filename == "pom.xml" or
            filename == "build.gradle" or
            filename == "Dockerfile" or
            filename == "requirements.txt" or
            filename == "README.md" or
            filename == "LICENSE" or
            filename.endswith('.ipynb') or
            filename.endswith('.txt') or
            filename.endswith('.csv')
        )
    )

def get_directory_and_files(path, exclude_content_files, prefix=''):
    structure = []
    all_files = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            if item not in ['venv', '__pycache__', '.git', 'dist', 'node_modules', 'gen.py', 'third_party', 'results', 'test']:  # 排除不需要显示的目录
                structure.append(prefix + '📁 ' + item)
                sub_structure, sub_files = get_directory_and_files(item_path, exclude_content_files, prefix + '    ')
                structure.extend(sub_structure)
                all_files.extend(sub_files)
        elif os.path.isfile(item_path) and is_code_file(item):
            structure.append(prefix + '📄 ' + item)
            all_files.append(item_path)
    return structure, all_files

# 指定要处理的目录路径
target_dir = './'

# 要省略内容的文件列表（相对路径）
exclude_content_files = []  # 示例文件名，可以填入需要省略内容的文件名列表

# 获取目录结构和所有代码文件
directory_structure, code_files = get_directory_and_files(target_dir, exclude_content_files)

# 将目录结构和代码文件内容合并到一个文件中, 用target_dir的最后一个名称来命名
filename = target_dir.split('/')[-1] + '_all_code.txt'
if filename == '_all_code.txt':  # 如果目标目录是当前目录，则改名
    filename = 'all_code.txt'

with open(filename, 'w', encoding='utf-8') as outfile:
    outfile.write("Here's the directory structure and code files:\n\n")

    # 写入目录结构
    outfile.write("Directory Structure:\n")
    for item in directory_structure:
        outfile.write(item + "\n")
    outfile.write("\n")

    # 写入代码文件内容
    outfile.write("Code Files:\n")

    # 配置展示策略
    txt_show_lines = 3
    csv_show_lines = 10
    csv_show_fields = 5

    for fname in code_files:
        relative_path = os.path.relpath(fname, target_dir)
        basename = os.path.basename(fname)
        outfile.write(f"----- Start of {relative_path} -----\n")
        
        if basename in exclude_content_files:
            # 如果在排除列表中，直接省略内容
            outfile.write(f"{relative_path} 内容省略\n")
        else:
            # 根据文件类型决定展示内容
            _, ext = os.path.splitext(basename)
            if ext == '.txt':
                # 对于txt文件：只读前三行，如超过三行提示部分内容
                with open(fname, 'r', encoding='utf-8', errors='replace') as infile:
                    lines = infile.readlines()
                    first_lines = lines[:txt_show_lines]
                    outfile.write(''.join(first_lines))
                    if len(lines) > txt_show_lines:
                        outfile.write("\n...（仅展示部分内容）...\n")
            
            elif ext == '.csv':
                # 对于csv文件：展示前10行，并对每行只展示前5个字段，多余字段用...表示
                # 如超过10行，则在结尾提示仍有更多内容
                with open(fname, 'r', encoding='utf-8', errors='replace') as infile:
                    lines = infile.readlines()
                    display_lines = lines[:csv_show_lines]

                    for line in display_lines:
                        fields = line.strip().split(',')
                        display_fields = fields[:csv_show_fields]
                        line_output = ','.join(display_fields)
                        if len(fields) > csv_show_fields:
                            line_output += "..."
                        outfile.write(line_output + "\n")

                    if len(lines) > csv_show_lines:
                        outfile.write("...（仅展示部分内容）...\n")

            else:
                # 对于其他类型文件：全部展示
                with open(fname, 'r', encoding='utf-8', errors='replace') as infile:
                    outfile.write(infile.read())

        outfile.write(f"\n----- End of {relative_path} -----\n\n")
print(f"文件已生成: {filename}")
