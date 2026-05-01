import os

with open("describe_files.md", "r") as f:
    appendix_content = f.read()

with open("neurips_submission_topic_a/main.tex", "r") as f:
    main_tex = f.read()

appendix_string = "\\newpage\n\\appendix\n\n" + appendix_content + "\n\n\\end{document}"

new_main = main_tex.replace("\\end{document}", appendix_string)

with open("neurips_submission_topic_a/main.tex", "w") as f:
    f.write(new_main)

print("Appendix successfully appended to main.tex.")
