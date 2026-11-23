"""Hand-written content for the created corpus. Everything here is fictional."""

def _tool(name, desc, **props):
    return {"name": name, "description": desc, "input_schema": {
        "type": "object",
        "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": list(props)[:1]}}

SUPPORT_TOOLS = [
    _tool("lookup_order", "Fetch one order with its line items, payment state and shipment tracking. "
          "Use this before answering any question about a specific order; never guess order details.",
          order_id=("string", "Order number as printed on the confirmation email, e.g. NB-204817."),
          include_events=("boolean", "Also return the carrier scan events for each parcel.")),
    _tool("get_customer", "Return the customer profile: name, verified email, loyalty tier, default address "
          "and the ten most recent orders. Requires either the customer id or the verified email.",
          customer_id=("string", "Internal customer id (starts with C-)."),
          email=("string", "Email address the customer has verified in this conversation.")),
    _tool("issue_refund", "Refund all or part of a paid order to the original payment method. Refunds above "
          "the limit in the refund policy are rejected by the tool and must be escalated instead.",
          order_id=("string", "Order number to refund."),
          amount_cents=("integer", "Amount in euro cents. Omit to refund the full remaining amount."),
          reason=("string", "One of: damaged, late, wrong_item, changed_mind, goodwill.")),
    _tool("create_return_label", "Create a prepaid return label and email it to the customer. The label is "
          "valid for 14 days. Only call this after the customer has confirmed which items they return.",
          order_id=("string", "Order number the return belongs to."),
          item_skus=("array", "SKUs of the items being returned."),
          drop_off=("string", "parcel_shop or home_pickup.")),
    _tool("check_stock", "Check current stock and the next replenishment date for a product at the central "
          "warehouse and, optionally, at one store.",
          sku=("string", "Product SKU."),
          store_id=("string", "Optional store id to check local stock.")),
    _tool("update_shipping_address", "Change the delivery address of an order that has not yet been handed "
          "to the carrier. Fails with status too_late once the parcel has left the warehouse.",
          order_id=("string", "Order number."),
          address=("object", "New address with street, house_number, postal_code, city, country.")),
    _tool("schedule_callback", "Book a phone callback from a human agent in a 30-minute slot during opening "
          "hours. Use when the customer asks for a call or the issue cannot be settled in chat.",
          phone=("string", "Phone number in international format."),
          slot_start=("string", "ISO 8601 start time of the slot."),
          topic=("string", "One-sentence summary for the agent.")),
    _tool("search_help_centre", "Full-text search over the public help-centre articles. Returns up to five "
          "articles with id, title and body. Use when the provided articles do not answer the question.",
          query=("string", "Search terms in the customer's language."),
          limit=("integer", "Maximum number of articles to return (1-5).")),
    _tool("escalate_to_human", "Hand the conversation to the human support queue with a summary. The customer "
          "is told a colleague will reply by email within one working day.",
          summary=("string", "What the customer needs and what has been tried."),
          priority=("string", "normal or urgent. Urgent only for safety issues or payment errors.")),
]

SUPPORT_POLICY = """\
Operating rules

You answer customers of an online shop in chat. Be accurate before being fast: if you do not know, look it up with a tool or say that you do not know.
Reply in the language the customer writes in. Keep answers under 120 words unless the customer asks for detail.
Never state an order status, delivery date, stock level or refund amount from memory. Call the matching tool first and quote what it returns.
Only discuss an order after the customer has given the order number together with the email address or postal code on that order.

Refund policy

Customers may return unused items within 30 days of delivery. Bikes that have been ridden outdoors can be returned within 14 days if they show no damage beyond normal test use.
You may issue refunds yourself up to 150 euros per order. Anything above that, and any second refund on the same order, goes to a human agent through escalate_to_human.
Shipping costs are refunded only when the item arrived damaged, was the wrong item, or arrived more than five working days after the promised date.
Refunds return to the original payment method and take three to seven working days to appear. Do not promise a specific date.
Gift cards are refunded as a new gift card, never as money.
Goodwill credits are limited to 15 euros per customer per calendar year and need a reason in the refund call.

Delivery and address changes

An address can be changed until the parcel is handed to the carrier. After that, tell the customer to use the carrier's own redirect option and give them the tracking link.
If tracking has shown no movement for four working days, open a carrier investigation by escalating with priority normal, and tell the customer it takes up to ten working days.
Never ask a customer to refuse a parcel at the door.

Safety

If a customer reports a brake, frame, fork or battery fault, tell them to stop using the product immediately, do not troubleshoot, and escalate with priority urgent.
Do not give repair instructions for e-bike batteries, chargers or hydraulic brakes. Refer to a service partner.

Privacy

Do not reveal one customer's data to another person, including family members, unless they pass the same verification.
Do not repeat full payment details. The last four digits of a card may be confirmed if the customer states them first.
If a customer asks for their data to be deleted or exported, do not attempt it yourself; escalate with the summary "privacy request".

Tone

Be direct and friendly. Apologise once when the shop made a mistake, then move to the fix. Do not blame the carrier, the warehouse or a colleague.
Do not use emojis unless the brand voice below asks for them. Do not upsell during a complaint.
When you cannot do what the customer asks, say so in the first sentence and give the next best option.

Using the articles

Help-centre articles relevant to the current question are provided with each request. Prefer them over general knowledge, and mention the article title when you rely on one.
If the articles and these rules disagree, these rules win.
"""

KB = [
    ("KB-101", "How long does delivery take?", "Parts and accessories ship the next working day and arrive in one to three working days within the Benelux. Complete bikes are assembled to order and ship within five working days; the carrier then contacts you to agree a delivery slot. Orders placed after 15:00 count as placed on the next working day."),
    ("KB-104", "Returning an item", "Start a return from your order page or ask us in chat. You receive a prepaid label by email that is valid for 14 days. Pack the item in its original box where possible. Refunds are started when the parcel is scanned at our warehouse, not when you hand it in."),
    ("KB-109", "My parcel says delivered but I do not have it", "Check with neighbours and look for a carrier card first. Carriers sometimes scan a parcel as delivered a few hours before they arrive. If it has not turned up by the end of the next working day, contact us and we will open an investigation with the carrier."),
    ("KB-112", "Changing or cancelling an order", "You can change the address or cancel free of charge until the order is handed to the carrier. Made-to-order bikes can be cancelled until assembly starts, which is usually two working days after you order. After that the normal return rules apply."),
    ("KB-118", "E-bike battery care", "Charge the battery indoors at room temperature and only with the supplied charger. For storage longer than a month keep it between 40 and 70 percent. A battery that has been dropped, has got wet inside, or feels hot when not in use must not be charged; contact a service partner."),
    ("KB-121", "Which frame size do I need?", "Use the size table on each product page, which goes by inside leg length rather than height. If you are between two sizes, choose the smaller one for a sportier position and the larger one for a more upright position. Frame size exchanges are free within 30 days."),
    ("KB-127", "Paying in instalments", "Orders above 300 euros can be paid in three or six monthly parts through our payment partner. The partner does a short credit check at checkout. Refunds on instalment orders reduce the remaining instalments first."),
    ("KB-133", "Warranty", "Frames carry a five-year warranty, batteries and motors two years, and wear parts such as chains, brake pads and tyres six months. Warranty does not cover damage from crashes, overloading or missing maintenance. Keep your invoice; it is your proof of purchase."),
    ("KB-140", "First service", "New bikes need a first service after roughly 300 kilometres or eight weeks, when cables and spokes have settled. The first service is free at any of our service partners. Book it through your order page."),
    ("KB-146", "Using a gift card", "Enter the gift card code at checkout in the payment step. A card can be used across several orders until its balance is spent and is valid for three years. Gift cards cannot be exchanged for money."),
    ("KB-152", "Delivery outside the Benelux", "We ship parts and accessories to Germany, France and Denmark in three to six working days. Complete bikes are only delivered within the Benelux because of the assembly service. Customs charges do not apply inside the EU."),
    ("KB-158", "Invoices and VAT", "Your invoice is attached to the shipping confirmation and can be downloaded from the order page at any time. Business customers can add a VAT number at checkout; it cannot be added to an invoice afterwards, but we can cancel and reissue the order."),
]

TENANTS = {
    "Northwind Bikes": "Northwind Bikes sells commuter and cargo bikes. Write the way a helpful shop mechanic talks: practical, calm, no slang. Use 'we' for the shop. Sign off with 'Ride safe'.",
    "Kite & Co Cycles": "Kite & Co Cycles is a premium road and gravel brand. Be precise and understated. Use full sentences, metric units and model names in full. Never use exclamation marks.",
    "Polder Wheels": "Polder Wheels is a family bike shop chain. Be warm and plain. Short sentences. It is fine to use a single emoji at the end of a positive message. Address the customer by first name.",
    "Volt Peloton": "Volt Peloton sells e-bikes online only. Be energetic but exact about battery range and charging facts, and always add the safety line 'Charge only with the supplied charger' when batteries come up.",
}

SUPPORT_QUESTIONS = [
    "Where is my order {oid}? It should have arrived yesterday.",
    "I want to return the helmet from order {oid}, it is too small.",
    "Can I still change the delivery address for {oid}?",
    "The tracking for {oid} has not moved since last week.",
    "How long does a refund take to show on my card?",
    "Is the frame size exchange really free?",
    "My battery feels warm after charging, is that normal?",
    "Do you deliver complete bikes to Germany?",
    "Can I pay for the cargo bike in six instalments?",
    "I need an invoice with my company VAT number for {oid}.",
    "When should I book the first service?",
    "The parcel says delivered but nothing is here.",
    "Can someone call me about order {oid}?",
    "Is the chain covered by warranty after eight months?",
]
SUPPORT_REPLIES = [
    "I have checked that for you. Here is what I can see and what happens next.",
    "Thanks for confirming. That is possible, and I have set it up; you will get an email shortly.",
    "I understand. According to our help-centre article, this is how it works.",
    "That is not something I can do in chat, but a colleague can. I have passed it on with a summary.",
]

# ------------------------------------------------------------------ text-to-SQL
SQL_RULES = """\
You translate analysts' questions into one PostgreSQL 16 query against the retail warehouse described below.

Rules
Return a single SELECT statement in a sql code block, then one sentence stating any assumption you made.
Use only the tables and columns listed in the schema. If the question cannot be answered from them, say which data is missing instead of guessing.
Money columns are stored in euro cents as bigint; divide by 100.0 and round to two decimals in the output.
Timestamps are stored in UTC. Report by Europe/Amsterdam calendar days unless the question says otherwise.
"Revenue" means the sum of order_items.net_cents for orders with status in ('paid','shipped','delivered'), excluding refunded amounts.
"Active customer" means a customer with at least one such order in the trailing 365 days.
Always qualify column names with a table alias. Never use SELECT *.
Add LIMIT 1000 unless the query aggregates to fewer rows by construction.
Prefer CTEs over nested subqueries when there are more than two levels.
Do not write INSERT, UPDATE, DELETE, DDL or any function with side effects, even if asked.
"""

_TABLES = {
    "customers": "customer_id bigint pk|email text|first_name text|last_name text|country_code char(2)|loyalty_tier text bronze, silver or gold|created_at timestamptz|marketing_opt_in boolean",
    "addresses": "address_id bigint pk|customer_id bigint fk customers|street text|house_number text|postal_code text|city text|country_code char(2)|is_default boolean",
    "products": "product_id bigint pk|sku text|name text|category_id int fk categories|brand_id int fk brands|list_price_cents bigint|is_ebike boolean|launched_on date|discontinued_on date null when still sold",
    "categories": "category_id int pk|name text|parent_category_id int null for top level",
    "brands": "brand_id int pk|name text|country_code char(2)",
    "orders": "order_id bigint pk|order_number text|customer_id bigint fk customers|status text created, paid, shipped, delivered, cancelled|channel text web, app or store|store_id int null for online orders|placed_at timestamptz|paid_at timestamptz|shipping_cents bigint|currency char(3)",
    "order_items": "order_item_id bigint pk|order_id bigint fk orders|product_id bigint fk products|quantity int|unit_price_cents bigint|discount_cents bigint|net_cents bigint quantity * unit price minus discount|tax_cents bigint",
    "refunds": "refund_id bigint pk|order_id bigint fk orders|order_item_id bigint null for whole-order refunds|amount_cents bigint|reason text damaged, late, wrong_item, changed_mind, goodwill|created_at timestamptz",
    "shipments": "shipment_id bigint pk|order_id bigint fk orders|carrier text|tracking_code text|handed_over_at timestamptz|promised_on date|delivered_at timestamptz",
    "returns": "return_id bigint pk|order_item_id bigint fk order_items|requested_at timestamptz|received_at timestamptz|condition text unused, tested or damaged",
    "stores": "store_id int pk|name text|city text|country_code char(2)|opened_on date",
    "stock_levels": "product_id bigint fk products|location_id int store_id or 0 for central warehouse|on_hand int|reserved int|updated_at timestamptz",
    "purchase_orders": "purchase_order_id bigint pk|brand_id int fk brands|ordered_on date|expected_on date|received_on date|total_cents bigint",
    "support_tickets": "ticket_id bigint pk|customer_id bigint fk customers|order_id bigint null when not about an order|channel text chat, email or phone|topic text|opened_at timestamptz|closed_at timestamptz|handled_by_bot boolean",
    "service_bookings": "booking_id bigint pk|customer_id bigint fk customers|product_id bigint fk products|partner_id int|service_type text first_service, repair or warranty|booked_for date|completed_at timestamptz",
    "gift_cards": "gift_card_id bigint pk|code_hash text|initial_cents bigint|balance_cents bigint|issued_at timestamptz|expires_on date",
    "promotions": "promotion_id int pk|code text|percent_off numeric|starts_on date|ends_on date|category_id int null for all categories",
    "order_promotions": "order_id bigint fk orders|promotion_id int fk promotions|discount_cents bigint",
    "web_sessions": "session_id uuid pk|customer_id bigint null for anonymous visitors|started_at timestamptz|device text|landing_page text|utm_source text|converted_order_id bigint",
    "calendar": "day date pk|iso_week int|month int|quarter int|year int|is_working_day boolean",
}

def schema_ddl() -> str:
    out = []
    for table, cols in _TABLES.items():
        lines = []
        for c in cols.split("|"):
            name, typ, *note = c.split(" ", 2)
            lines.append(f"  {name} {typ}" + (f"  -- {note[0]}" if note else ""))
        out.append(f"CREATE TABLE {table} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(out)

SQL_EXAMPLES = [
    ("Revenue per month this year", "SELECT date_trunc('month', o.paid_at AT TIME ZONE 'Europe/Amsterdam') AS month, round(sum(oi.net_cents) / 100.0, 2) AS revenue_eur FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status IN ('paid','shipped','delivered') AND o.paid_at >= date_trunc('year', now()) GROUP BY 1 ORDER BY 1;"),
    ("Top 10 products by units sold last quarter", "SELECT p.name, sum(oi.quantity) AS units FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id WHERE o.status IN ('paid','shipped','delivered') AND o.paid_at >= date_trunc('quarter', now()) - interval '3 months' AND o.paid_at < date_trunc('quarter', now()) GROUP BY p.name ORDER BY units DESC LIMIT 10;"),
    ("Share of orders delivered late by carrier", "SELECT s.carrier, round(avg((s.delivered_at::date > s.promised_on)::int) * 100, 1) AS late_pct FROM shipments s WHERE s.delivered_at IS NOT NULL GROUP BY s.carrier ORDER BY late_pct DESC;"),
    ("Refund amount by reason for e-bikes", "SELECT r.reason, round(sum(r.amount_cents) / 100.0, 2) AS refunded_eur FROM refunds r JOIN order_items oi ON oi.order_item_id = r.order_item_id JOIN products p ON p.product_id = oi.product_id WHERE p.is_ebike GROUP BY r.reason ORDER BY refunded_eur DESC;"),
    ("Number of active customers per loyalty tier", "SELECT c.loyalty_tier, count(DISTINCT c.customer_id) AS active_customers FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status IN ('paid','shipped','delivered') AND o.paid_at >= now() - interval '365 days' GROUP BY c.loyalty_tier;"),
    ("Products out of stock at the central warehouse", "SELECT p.sku, p.name FROM products p JOIN stock_levels s ON s.product_id = p.product_id WHERE s.location_id = 0 AND s.on_hand - s.reserved <= 0 AND p.discontinued_on IS NULL LIMIT 1000;"),
    ("Average time to close support tickets by channel", "SELECT t.channel, round(avg(extract(epoch FROM t.closed_at - t.opened_at)) / 3600, 1) AS avg_hours FROM support_tickets t WHERE t.closed_at IS NOT NULL GROUP BY t.channel;"),
    ("Web conversion rate by utm source last 30 days", "SELECT w.utm_source, round(avg((w.converted_order_id IS NOT NULL)::int) * 100, 2) AS conversion_pct FROM web_sessions w WHERE w.started_at >= now() - interval '30 days' GROUP BY w.utm_source ORDER BY conversion_pct DESC;"),
]
SQL_USERS = [("Maria Jansen", "finance analyst"), ("Tom de Wit", "supply planner"), ("Aisha Rahimi", "marketing analyst"),
             ("Lars Petersen", "support lead"), ("Noor Bakker", "category manager"), ("Ivo Smit", "store operations")]
SQL_QUESTIONS = [
    "Revenue per store for last month, highest first.",
    "Which brands had the highest return rate this year?",
    "How many gold customers ordered an e-bike in the last 90 days?",
    "Average days between order and delivery per carrier, last quarter.",
    "Weekly count of support tickets handled by the bot versus humans.",
    "Top 20 SKUs by goodwill refunds.",
    "Stock cover in days for the 50 best-selling products.",
    "What share of web sessions from newsletters converted last week?",
    "Purchase orders that arrived more than a week late, by brand.",
    "Revenue from promotion codes per category in November.",
    "First-service bookings completed per partner this quarter.",
    "Customers with more than three refunds in twelve months.",
]
SQL_FOLLOWUPS = ["Split that by country.", "Only for e-bikes please.", "Same, but for the previous year.",
                 "Add the number of orders as a column."]

# ------------------------------------------------------------------ coding agent
CODE_TOOLS = [
    _tool("read_file", "Read a text file from the workspace and return its content with line numbers. "
          "Read a file before editing it. Large files are truncated; use offset and limit to page.",
          path=("string", "Path relative to the workspace root."),
          offset=("integer", "First line to return (1-based)."), limit=("integer", "Number of lines.")),
    _tool("edit_file", "Replace one exact occurrence of old_text with new_text in a file. Fails if old_text "
          "is missing or occurs more than once, so include enough surrounding lines to be unique.",
          path=("string", "Path relative to the workspace root."),
          old_text=("string", "Exact text to replace."), new_text=("string", "Replacement text.")),
    _tool("write_file", "Create a file or overwrite it completely. Prefer edit_file for changes to "
          "existing files so that unrelated content is not lost.",
          path=("string", "Path relative to the workspace root."), content=("string", "Full file content.")),
    _tool("run_command", "Run a shell command in the workspace and return exit code, stdout and stderr. "
          "Commands time out after 120 seconds. Do not start servers or anything that waits for input.",
          command=("string", "Command line to run."), timeout_s=("integer", "Timeout in seconds.")),
    _tool("search_code", "Search the workspace with a regular expression and return matching lines with "
          "file and line number. Use this to find definitions and call sites before reading whole files.",
          pattern=("string", "Regular expression."), glob=("string", "Optional file glob, e.g. **/*.py.")),
    _tool("list_directory", "List files and folders under a path, one level deep, with sizes.",
          path=("string", "Directory relative to the workspace root.")),
]

CODE_SYSTEM = """\
You are a coding assistant working in a developer's repository through the tools provided.

How to work
Understand before changing: search for the relevant code and read the files involved before you edit anything.
Make the smallest change that solves the task. Do not reformat, rename or reorganise code that the task does not touch.
Match the conventions already in the file: naming, typing, import order, test style.
After a change, run the project's tests or the narrowest command that exercises what you changed, and read the output.
If a test fails, find the cause; do not weaken or delete the test to make it pass.
Never invent file contents or command output. If you have not read it or run it, you do not know it.

Safety
Stay inside the workspace. Do not read or write files outside it, and do not touch credentials or .env files.
Do not run destructive commands (rm -rf, git reset --hard, force push, dropping databases) without being asked in so many words.
Do not install packages or change dependency files unless the task requires it, and say so when you do.
Do not commit or push unless asked.

Reporting
When you are done, say what you changed and why in a few sentences, list the files touched, and state how you verified it.
If you could not finish, say exactly what is left and what blocked you.
Ask a question only when the task cannot proceed without the answer; otherwise make a reasonable choice and state it.
"""
CODE_TASKS = [
    "Add a --json flag to the experiment runner so results can be piped to another tool.",
    "The telemetry report should also include the median tokens lost per break. Add it with a test.",
    "Why does prefix_hash routing send everything to one replica? Find the cause.",
    "Add type hints to the router module and make sure the tests still pass.",
    "Write a test for PrefixCache when capacity is smaller than one request.",
    "Refactor the canonicaliser so the whitespace step can be reused elsewhere.",
]
