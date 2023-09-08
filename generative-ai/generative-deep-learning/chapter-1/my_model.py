
print("hello docker terminal!")

# When running in docker redirect stdout to a log file
import sys
sys.stdout = open("/app/output.log", "w")


print("hello docker logs!")
