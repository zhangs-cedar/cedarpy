from init import *


class ConfigEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("配置文件编辑器")

        # 左侧 Treeview
        self.tree = ttk.Treeview(root, show="tree")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # 右侧容器
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 右侧上半部分：说明文档
        self.doc_label = tk.Label(self.right_frame, text="说明文档:", font=("Arial", 12))
        self.doc_label.pack(fill=tk.X)
        self.doc_text = tk.Text(self.right_frame, width=50, height=10)
        self.doc_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.doc_text.insert(tk.END, "请选择一个配置项以查看说明信息。")

        # 右侧下半部分：配置值输入框和确认修改按钮
        self.value_label = tk.Label(self.right_frame, text="配置值:", font=("Arial", 12))
        self.value_label.pack(fill=tk.X)
        self.value_entry = tk.Entry(self.right_frame, width=50)
        self.value_entry.pack(fill=tk.X, pady=5)
        self.save_button = tk.Button(self.right_frame, text="确认修改", command=self.save_value)
        self.save_button.pack(fill=tk.X, pady=5)

        # self.config_data = config_data
        # 模拟的多层树状数据
        self.config_data = {
            "Server": {"IP": {"value": "192.168.1.1", "doc": "服务器的IP地址"}, "Port": {"value": 8080, "doc": "服务器的端口号"}},
            "User": {"Username": {"value": "admin", "doc": "用户的用户名"}, "Password": {"value": "password123", "doc": "用户的密码"}},
            "Logging": {"Level": {"value": "DEBUG", "doc": "日志级别"}, "File": {"value": "log.txt", "doc": "日志文件路径"}},
            "Features": {
                "Feature1": {
                    "Enabled": {"value": "True", "doc": "是否启用Feature1"},
                    "Description": {"value": "This is feature 1", "doc": "Feature1的描述信息"},
                },
                "Feature2": {
                    "Enabled": {"value": "False", "doc": "是否启用Feature2"},
                    "Description": {"value": "This is feature 2", "doc": "Feature2的描述信息"},
                },
            },
        }

        # 当前选中的配置项
        self.current_key = None
        self.current_section = None

        # 加载模拟数据
        self.load_simulated_data()

    def load_simulated_data(self):
        """加载模拟的树状数据"""
        self.tree.delete(*self.tree.get_children())  # 清空树
        for section, items in self.config_data.items():
            section_node = self.tree.insert("", "end", text=section)
            if isinstance(items, dict):
                for key, value_dict in items.items():
                    if isinstance(value_dict, dict):
                        if "value" in value_dict:  # 直接配置项
                            self.tree.insert(section_node, "end", text=key, values=value_dict["value"])
                        else:  # 嵌套配置项
                            sub_section_node = self.tree.insert(section_node, "end", text=key)
                            for sub_key, sub_value_dict in value_dict.items():
                                self.tree.insert(sub_section_node, "end", text=sub_key, values=sub_value_dict["value"])
        self.tree.selection_set(self.tree.get_children()[0])  # 默认选中第一个节点
        self.on_select(None)

    def on_select(self, event):
        """树节点选择事件"""
        selected_item = self.tree.selection()
        if selected_item:
            item = self.tree.item(selected_item)
            if "values" in item and item["values"]:  # 检查 values 是否存在且不为空
                value = item["values"][0]
                self.value_entry.delete(0, tk.END)
                self.value_entry.insert(tk.END, value)

                # 获取当前选中的配置项路径
                path = self.get_item_path(selected_item)
                if len(path) == 2:  # 直接配置项
                    self.current_section, self.current_key = path
                    self.doc_text.delete(1.0, tk.END)
                    self.doc_text.insert(tk.END, self.config_data[self.current_section][self.current_key]["doc"])
                elif len(path) == 3:  # 嵌套配置项
                    self.current_section, sub_section, self.current_key = path
                    self.doc_text.delete(1.0, tk.END)
                    self.doc_text.insert(tk.END, self.config_data[self.current_section][sub_section][self.current_key]["doc"])
            else:
                self.value_entry.delete(0, tk.END)  # 如果没有值，则清空输入框
                self.doc_text.delete(1.0, tk.END)  # 清空说明文档

    def get_item_path(self, item):
        """获取树节点的路径"""
        path = []
        while item:
            path.insert(0, self.tree.item(item, "text"))
            item = self.tree.parent(item)
        return path

    def save_value(self):
        """保存配置值"""
        if self.current_key and self.current_section:
            new_value = self.value_entry.get()
            if len(self.get_item_path(self.tree.selection())) == 2:
                self.config_data[self.current_section][self.current_key]["value"] = new_value
            else:
                self.current_section, sub_section, self.current_key = self.get_item_path(self.tree.selection())
                self.config_data[self.current_section][sub_section][self.current_key]["value"] = new_value
            messagebox.showinfo("成功", "配置已更新")
        else:
            messagebox.showwarning("警告", "请选择一个有效的配置项")

    def open_config(self):
        """打开配置文件"""
        file_path = filedialog.askopenfilename(filetypes=[("配置文件", "*.conf"), ("所有文件", "*.*")])
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.config_data = {}
                current_section = None
                for line in file:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1]
                        self.config_data[current_section] = {}
                    elif "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if current_section:
                            self.config_data[current_section][key] = {"value": value, "doc": ""}
            self.load_simulated_data()

        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件：{e}")

    def save_config(self):
        """保存配置文件"""
        file_path = filedialog.asksaveasfilename(filetypes=[("配置文件", "*.conf"), ("所有文件", "*.*")])
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                for section, items in self.config_data.items():
                    file.write(f"[{section}]\n")
                    for key, value_dict in items.items():
                        file.write(f"{key}={value_dict['value']}\n")
                    file.write("\n")
            messagebox.showinfo("成功", "配置文件已保存")
        except Exception as e:
            messagebox.showerror("错误", f"无法保存文件：{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditor(root)
    root.mainloop()
