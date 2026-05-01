import os

with open("describe_files.md", "r") as f:
    content = f.read()

# Replace paths to use the new folder
content = content.replace("plots_a/", "graphs__std_a/")

# Give the new section a distinct title and label
content = content.replace(
    "\\section{Extended Visualization Suite and Statistical Bounds}\n\\label{appendix:extended_visuals}",
    "\\section{Visualization Suite with Visualized Variance Bounds}\n\\label{appendix:std_visuals}"
)

# Modify the preamble to reflect the visible variance bands
content = content.replace(
    "To optimize trace readability and visual clarity in the graphical plots, shaded variance bounds—which inherently overlap significantly during mapping collapses—have been excluded from the primary axes.",
    "Unlike Appendix \\ref{appendix:extended_visuals}, the following figures explicitly visualize the standard deviation variance bounds directly on the primary axes (via shaded regions or error bars). Note that the high-variance regions (e.g., mapping collapses) may cause significant visual trace overlap."
)
content = content.replace(
    "Instead, we report the overarching stability metrics associated with every evaluated configuration mathematically within the corresponding figure captions. ",
    "The overarching mathematical stability metrics remain documented within the corresponding figure captions for exact numerical reference alongside the visual bands. "
)

with open("neurips_submission_topic_a/main.tex", "r") as f:
    main_tex = f.read()

appendix_string = "\\newpage\n" + content + "\n\n\\end{document}"

new_main = main_tex.replace("\\end{document}", appendix_string)

with open("neurips_submission_topic_a/main.tex", "w") as f:
    f.write(new_main)

print("Second appendix with STD graphs successfully appended!")
