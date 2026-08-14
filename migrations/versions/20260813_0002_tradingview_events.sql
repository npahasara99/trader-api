-- Monitoring-only TradingView alerts. These rows never represent executable orders.
create table if not exists tradingview_signal_events (
    id bigserial primary key,
    event_id varchar(80) not null unique,
    ticker varchar(32) not null,
    timeframe varchar(16) not null,
    event_type varchar(40) not null,
    price double precision not null,
    occurred_at timestamptz not null,
    received_at timestamptz not null default now(),
    indicators_json text,
    payload_json text not null,
    processed boolean not null default false,
    processing_status varchar(40) not null default 'pending_replan',
    re_evaluation_requested boolean not null default true,
    execution_requested boolean not null default false
);

create index if not exists ix_tradingview_signal_events_event_id
    on tradingview_signal_events (event_id);
create index if not exists ix_tradingview_signal_events_ticker
    on tradingview_signal_events (ticker);
create index if not exists ix_tradingview_signal_events_monitor
    on tradingview_signal_events (ticker, processed, received_at);
