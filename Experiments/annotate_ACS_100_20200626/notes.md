upload to willow

    scp * rodriguezne2@172.18.233.71:/home/share/projects/sbks/annotation_framework/data


The algorithm chooses for a span either the highest scoring annotation, or the longest spanning annotation. I haven't settled on a best approach yet. The longest span seems to work well for resolve chemical and gene overlaps, but not for species (as you can see above). If two spans overlap exactly, the one with the highest score prevails. Using that approach, Linneaus seems to be more most common in the results.

Longest span helps in the case that chemical and gene overlap where there's a mention of a gene/protein that acts on a chemical. There, we'd want to keep gene.