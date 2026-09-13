import sys

def append_business():
    with open("app (5).py", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Python lines are 0-indexed, so line 943 in text editor is index 942.
    # Lines 942 (which is # --- ANALISIS...) to 1338 (index 1337)
    block1 = lines[941:1338]
    
    # Lines 1412 (index 1411: # --- Matching indexado ---) to 1615 (index 1614)
    block2 = lines[1411:1615]
    
    with open("core/business.py", "a", encoding="utf-8") as out:
        out.write("\n\n")
        out.write("".join(block1))
        out.write("\n\n")
        out.write("".join(block2))
        
    print("Code blocks appended to core/business.py")

if __name__ == "__main__":
    append_business()
