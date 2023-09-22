.PHONY: clean

clean:
	@for ext in out err; do \
	    for name in $$(ls *.$$ext 2>/dev/null | cut -d'.' -f1 | sort | uniq); do \
	        max_id=$$(ls $$name.*.$$ext 2>/dev/null | cut -d'.' -f2 | sort -nr | head -1); \
	        for file in $$name.*.$$ext; do \
	            id=$$(echo $$file | cut -d'.' -f2); \
	            if [ "$$id" -lt "$$max_id" ]; then \
	                rm -f $$file; \
	            fi \
	        done \
	    done \
	done

