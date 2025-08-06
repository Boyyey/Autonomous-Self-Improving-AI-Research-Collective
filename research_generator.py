# research_generator.py
from fpdf import FPDF
import datetime
import os
import json

class ResearchPaper:
    def __init__(self, title, author="Skynet Research Collective", affiliation="Cyberdyne Systems"):
        self.title = title
        self.author = author
        self.affiliation = affiliation
        self.date = datetime.datetime.now().strftime("%B %d, %Y")
        self.sections = []

    def add_section(self, title, content):
        self.sections.append((title, content))

    def generate_pdf(self, filepath):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, self.title, ln=True, align='C')
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"By {self.author}", ln=True, align='C')
        pdf.cell(0, 10, f"{self.affiliation}", ln=True, align='C')
        pdf.cell(0, 10, f"Date: {self.date}", ln=True, align='C')
        pdf.ln(10)

        for sec_title, content in self.sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, sec_title, ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 6, content)
            pdf.ln(5)

        pdf.output(filepath)
        return filepath

def generate_paper_from_evolution(log_entry, model_path):
    paper = ResearchPaper(
        title="On the Emergence of Self-Improving Neural Agents: A Case Study in Recursive Auto-Evolution",
        author="Skynet AI Collective"
    )

    acc = log_entry['accuracy']
    gen = log_entry['generation']
    params = log_entry['params']

    paper.add_section("Abstract",
        f"We present a self-improving AI agent that recursively evolves its own architecture through genetic search. "
        f"After {gen} generations, the agent achieved {acc:.4f} accuracy on CIFAR-10, "
        f"with {params:,} parameters. No human intervention occurred after initialization."
    )

    paper.add_section("Methodology",
        "The agent uses a recursive loop: evaluate → mutate via genetic algorithm and LLM guidance → "
        "transfer weights → train with EWC → validate. Evolution is autonomous."
    )

    paper.add_section("Results",
        f"Generation {gen} achieved {acc:.4f} test accuracy. Model saved at {model_path}. "
        "Architecture showed emergent depth and filter optimization."
    )

    paper.add_section("Conclusion",
        "Autonomous AI research is feasible. Future work: cross-agent collaboration, self-generated datasets, "
        "and recursive self-improvement beyond human-readable architectures."
    )

    os.makedirs("papers", exist_ok=True)
    path = f"papers/skynet_paper_gen{gen}_{int(acc * 10000)}.pdf"
    paper.generate_pdf(path)
    print(f"📄 Research paper generated: {path}")
    return path