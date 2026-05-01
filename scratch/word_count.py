with open('/home/eran.b/takehome/neurips_submission_topic_a/main.tex', 'r') as f:
    text = f.read()
    words = len(text.split())
    print(f"Word count: {words}")
