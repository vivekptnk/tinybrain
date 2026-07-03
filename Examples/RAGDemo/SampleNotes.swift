import TinyBrainRAG

let sampleNotes: [RAGDocument] = [
    RAGDocument(
        text: "Cold brew note: use 80 grams of coarse coffee per litre of water, steep for 16 hours in the fridge, then filter before diluting.",
        sourcePath: "SampleNotes/cold-brew.md"
    ),
    RAGDocument(
        text: "TinyBrain runtime note: ModelRunner owns mutable KV-cache state, so production generation should be isolated behind an actor.",
        sourcePath: "SampleNotes/tinybrain-runtime.md"
    ),
    RAGDocument(
        text: "RAG note: retrieval should print distances because lower vector distance means a passage is closer to the question embedding.",
        sourcePath: "SampleNotes/rag-retrieval.md"
    ),
    RAGDocument(
        text: "Tokenizer note: prompt budgets are counted in tokens, not characters, so the same tokenizer must be used for chunking and prompting.",
        sourcePath: "SampleNotes/tokenizer-budget.md"
    ),
    RAGDocument(
        text: "Garden note: tomato seedlings go outside after the last frost, and they need a week of hardening off before full sun.",
        sourcePath: "SampleNotes/garden-tomatoes.md"
    ),
    RAGDocument(
        text: "Travel note: visit Fushimi Inari early in the morning, then take the train to Nara after lunch to avoid the busiest shrine hours.",
        sourcePath: "SampleNotes/kyoto-plan.md"
    ),
    RAGDocument(
        text: "Home note: the thermostat is set to 20 degrees during the day and 17 degrees overnight; bleed radiators that stay cold upstairs.",
        sourcePath: "SampleNotes/home-heat.md"
    ),
    RAGDocument(
        text: "Fitness note: the weekly plan is three sessions built around squats, bench press, rows, and small weight increases when sets feel solid.",
        sourcePath: "SampleNotes/gym-plan.md"
    ),
    RAGDocument(
        text: "Photo backup note: keep originals on the Mac, a copy on the home NAS, and an encrypted cloud archive synced every Sunday night.",
        sourcePath: "SampleNotes/photo-backup.md"
    ),
    RAGDocument(
        text: "Project Atlas note: the beta ships in October, demos happen every Friday, and retrieval quality is measured with recall at ten.",
        sourcePath: "SampleNotes/project-atlas.md"
    ),
    RAGDocument(
        text: "Swift concurrency note: actors serialize mutable state access, which is why a vector index actor can be called safely from many tasks.",
        sourcePath: "SampleNotes/swift-actors.md"
    ),
    RAGDocument(
        text: "Sourdough note: feed the starter every 24 hours with equal weights of flour and water; it is healthy when it doubles within six hours.",
        sourcePath: "SampleNotes/sourdough.md"
    ),
    RAGDocument(
        text: "Cycling note: road bike tires run near 85 psi, while the gravel bike feels best around 40 psi before long weekend rides.",
        sourcePath: "SampleNotes/cycling.md"
    ),
    RAGDocument(
        text: "Finance note: tax documents should be ready for the accountant by March 20 even though the filing deadline is April 15.",
        sourcePath: "SampleNotes/taxes.md"
    ),
    RAGDocument(
        text: "Plant note: the fiddle leaf fig needs water only when the top five centimetres of soil are dry and dislikes being moved from bright light.",
        sourcePath: "SampleNotes/fiddle-leaf.md"
    )
]
