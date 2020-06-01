#!/usr/bin/perl

# TO RUN: perl huner2brat.pl <filename> <entitytype>

my $file = shift;
my $entitytype = shift;

$file=~/(.*?)\.conll.test/;
my $name = $1;

my $txtfile = $name . ".txt";
my $annfile = $name . ".ann";

open(TXT, ">$txtfile") || die "Could not open TXT file ($txtfile)\n";
open(ANN, ">$annfile") || die "Could not open ANN file ($annfile)\n";

open(FILE, $file) || die "Could not open CONLL file ($file)\n";

my $sentence = "";
my %annotations = ();
my $char = 0;
my $num = 0;

my $entity = "";
my $bspan = "";
my $espan = ""; 
my $mention = ""; 

while(<FILE>) {
    chomp;

    if($_=~/DOCSTART/) {
	my $blank = <FILE>;
	next;
    }
    
    #if blank line then add a new line character and increase char
    if($_=~/^\s*$/) {
	$sentence .= "\n";
	$char++;
	next; 
    }

    # otherwise get the term with the label
    my ($term, $pos, $label) = split/\s+/;

    #  turning the label into the entity label
    if($label ne "O") { $label = $entitytype; }
    
    # save the previous character span
    my $pspan = $char; 

    #  get the current character span
    @chars = split//, $term;
    $char += $#chars + 2;
    
    #  correct for end punctuation and commas
    if($term=~/^[\,\.]$/) { 
	chop $sentence; 
	$char--; 
    }
    $sentence .= "$term ";

    #if the term is an entity
    if($label ne "O") {
	# if the term is part of the previous entity
	if($entity eq $label) {
	    $espan = $char;
	    $mention .= " $term";
	}
	# otherwise it is the begining of its own entity
	elsif($entity ne $label) {
	    $entity = $label;
	    $bspan = $pspan;
	    $espan = $char;
	    $mention = $term;
	}
    }
    # if the term is not an entity
    else {
	# if the previous word was the end of an entity
	if($entity ne "")
	{
	    $espan--; 
	    push @annotations, "T$num\t$entity $bspan $espan\t$mention";
	    $num++; 
	}
	$entity = "";
	$bspan = "";
	$espan = "";
	$mention = "";
	
    }
}

foreach my $annot (@annotations) { 
	print ANN "$annot\n";
}

print TXT "$sentence\n"; 

close ANN; 
close TXT; 