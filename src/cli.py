"""Command-line interface for the NBA betting model."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
from datetime import date

from .db.session import init_db, get_session
from .db.models import Player, Team, PlayerGameLog, TeamSeasonStats
from .data.collectors.player_stats import PlayerStatsCollector
from .data.sources.nba_api_source import get_current_season

app = typer.Typer(help="NBA Player Props Betting Model")
console = Console()


@app.command()
def init():
    """Initialize the database."""
    console.print("[bold blue]Initializing database...[/bold blue]")
    init_db()
    console.print("[bold green]Database initialized successfully![/bold green]")


@app.command()
def sync_teams():
    """Sync NBA teams to the database."""
    with get_session() as session:
        collector = PlayerStatsCollector()
        with console.status("[bold blue]Syncing teams..."):
            teams = collector.sync_teams(session)
        console.print(f"[bold green]Synced {len(teams)} teams[/bold green]")


@app.command()
def sync_players():
    """Sync active NBA players to the database."""
    with get_session() as session:
        collector = PlayerStatsCollector()
        with console.status("[bold blue]Syncing players..."):
            players = collector.sync_players(session)
        console.print(f"[bold green]Synced {len(players)} players[/bold green]")


@app.command()
def backfill(
    season: Optional[str] = typer.Option(None, help="Season to backfill (e.g., 2024-25)"),
    player_name: Optional[str] = typer.Option(None, help="Specific player name to backfill"),
    limit: int = typer.Option(50, help="Max number of players to process")
):
    """Backfill historical player game logs."""
    if season is None:
        season = get_current_season()

    console.print(f"[bold blue]Backfilling data for season {season}...[/bold blue]")

    # Initialize database
    init_db()

    with get_session() as session:
        collector = PlayerStatsCollector()

        # First sync teams and players
        with console.status("[bold blue]Syncing teams..."):
            collector.sync_teams(session)

        with console.status("[bold blue]Syncing players..."):
            collector.sync_players(session)

        # Get players to process
        if player_name:
            from sqlalchemy import select
            players = session.execute(
                select(Player).where(Player.name.ilike(f"%{player_name}%"))
            ).scalars().all()
        else:
            from sqlalchemy import select
            players = session.execute(
                select(Player).where(Player.is_active == True).limit(limit)
            ).scalars().all()

        if not players:
            console.print("[yellow]No players found[/yellow]")
            return

        console.print(f"Processing {len(players)} players...")

        total_logs = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Collecting game logs...", total=len(players))

            for player in players:
                progress.update(task, description=f"Processing {player.name}...")
                try:
                    added = collector.collect_player_game_logs(
                        player.nba_api_id, season, session
                    )
                    total_logs += added
                except Exception as e:
                    console.print(f"[red]Error for {player.name}: {e}[/red]")
                progress.advance(task)

        console.print(f"[bold green]Added {total_logs} game logs[/bold green]")


@app.command()
def sync_team_stats(
    season: Optional[str] = typer.Option(None, help="Season (e.g., 2024-25)")
):
    """Sync team statistics (pace, ratings)."""
    if season is None:
        season = get_current_season()

    with get_session() as session:
        collector = PlayerStatsCollector()
        from sqlalchemy import select

        teams = session.execute(select(Team)).scalars().all()

        if not teams:
            console.print("[yellow]No teams found. Run sync-teams first.[/yellow]")
            return

        console.print(f"Syncing stats for {len(teams)} teams...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Syncing team stats...", total=len(teams))

            for team in teams:
                progress.update(task, description=f"Processing {team.abbreviation}...")
                try:
                    collector.collect_team_stats(team.nba_api_id, season, session)
                except Exception as e:
                    console.print(f"[red]Error for {team.abbreviation}: {e}[/red]")
                progress.advance(task)

        console.print("[bold green]Team stats synced![/bold green]")


@app.command()
def stats(player_name: str):
    """Show recent stats for a player."""
    with get_session() as session:
        from sqlalchemy import select

        player = session.execute(
            select(Player).where(Player.name.ilike(f"%{player_name}%"))
        ).scalar_one_or_none()

        if not player:
            console.print(f"[red]Player '{player_name}' not found[/red]")
            return

        logs = session.execute(
            select(PlayerGameLog)
            .where(PlayerGameLog.player_id == player.id)
            .order_by(PlayerGameLog.game_date.desc())
            .limit(10)
        ).scalars().all()

        if not logs:
            console.print(f"[yellow]No game logs found for {player.name}[/yellow]")
            return

        # Create table
        table = Table(title=f"{player.name} - Last 10 Games")
        table.add_column("Date", style="cyan")
        table.add_column("PTS", justify="right")
        table.add_column("REB", justify="right")
        table.add_column("AST", justify="right")
        table.add_column("STL", justify="right")
        table.add_column("BLK", justify="right")
        table.add_column("3PM", justify="right")
        table.add_column("MIN", justify="right")

        for log in logs:
            table.add_row(
                str(log.game_date),
                str(log.points),
                str(log.rebounds),
                str(log.assists),
                str(log.steals),
                str(log.blocks),
                str(log.fg3m),
                f"{log.minutes:.1f}" if log.minutes else "-"
            )

        console.print(table)

        # Show averages
        import statistics
        pts_avg = statistics.mean(log.points for log in logs if log.points)
        reb_avg = statistics.mean(log.rebounds for log in logs if log.rebounds)
        ast_avg = statistics.mean(log.assists for log in logs if log.assists)

        console.print(f"\n[bold]L10 Averages:[/bold] {pts_avg:.1f} PTS / {reb_avg:.1f} REB / {ast_avg:.1f} AST")


@app.command()
def list_players(
    limit: int = typer.Option(20, help="Number of players to show")
):
    """List players in the database."""
    with get_session() as session:
        from sqlalchemy import select, func

        # Get players with game counts
        results = session.execute(
            select(
                Player,
                func.count(PlayerGameLog.id).label('game_count')
            )
            .outerjoin(PlayerGameLog)
            .group_by(Player.id)
            .order_by(func.count(PlayerGameLog.id).desc())
            .limit(limit)
        ).all()

        table = Table(title="Players with Most Game Logs")
        table.add_column("Name", style="cyan")
        table.add_column("Games", justify="right")
        table.add_column("Active", justify="center")

        for player, game_count in results:
            table.add_row(
                player.name,
                str(game_count),
                "Yes" if player.is_active else "No"
            )

        console.print(table)


@app.command()
def project(
    player_name: str,
    prop: str = typer.Option("points", help="Prop type: points, rebounds, assists, steals, blocks, threes"),
    line: Optional[float] = typer.Option(None, help="Line to evaluate (e.g., 25.5)"),
    over_odds: int = typer.Option(-110, help="Over odds (American)"),
    under_odds: int = typer.Option(-110, help="Under odds (American)")
):
    """Generate projection for a player prop."""
    import numpy as np
    from sqlalchemy import select

    from .config.constants import PropType, PROP_THRESHOLDS
    from .models.statistical.negative_binomial import NegativeBinomialModel
    from .models.base import PlayerContext
    from .betting.ev_calculator import EVCalculator
    from .odds.devig import OddsConverter

    # Map prop string to PropType
    prop_map = {
        "points": PropType.POINTS,
        "rebounds": PropType.REBOUNDS,
        "assists": PropType.ASSISTS,
        "steals": PropType.STEALS,
        "blocks": PropType.BLOCKS,
        "threes": PropType.THREES,
        "turnovers": PropType.TURNOVERS
    }

    prop_type = prop_map.get(prop.lower())
    if not prop_type:
        console.print(f"[red]Unknown prop type: {prop}. Use: {list(prop_map.keys())}[/red]")
        return

    with get_session() as session:
        # Find player
        player = session.execute(
            select(Player).where(Player.name.ilike(f"%{player_name}%"))
        ).scalar_one_or_none()

        if not player:
            console.print(f"[red]Player '{player_name}' not found[/red]")
            return

        # Get game logs
        logs = session.execute(
            select(PlayerGameLog)
            .where(PlayerGameLog.player_id == player.id)
            .order_by(PlayerGameLog.game_date.desc())
        ).scalars().all()

        if len(logs) < 5:
            console.print(f"[yellow]Not enough data for {player.name} ({len(logs)} games)[/yellow]")
            return

        # Extract stat values
        stat_attr = prop_type.value
        if stat_attr == "threes":
            stat_attr = "fg3m"

        values = np.array([getattr(log, stat_attr) or 0 for log in logs])

        # Fit model
        model = NegativeBinomialModel(prop_type)
        model.fit(values)

        # Generate prediction
        context = PlayerContext(
            player_id=player.id,
            is_home=True,  # Default assumption
            rest_days=1
        )
        prediction = model.predict(context)

        # Display results
        console.print(f"\n[bold cyan]{player.name} - {prop_type.value.upper()} Projection[/bold cyan]")
        console.print(f"Based on {len(logs)} games\n")

        console.print(f"[bold]Predicted:[/bold] {prediction.predicted_mean:.1f} +/- {prediction.predicted_std:.1f}")
        console.print(f"[bold]Percentiles:[/bold] 25th={prediction.get_percentile(25):.1f}, 50th={prediction.get_percentile(50):.1f}, 75th={prediction.get_percentile(75):.1f}")

        # Show threshold probabilities
        thresholds = PROP_THRESHOLDS.get(prop_type, [])
        if thresholds:
            console.print(f"\n[bold]Over Probabilities:[/bold]")
            table = Table(show_header=True)
            table.add_column("Line", justify="right")
            table.add_column("P(Over)", justify="right")
            table.add_column("Fair Odds", justify="right")

            for thresh in thresholds:
                prob = prediction.prob_over(thresh)
                if prob > 0.01 and prob < 0.99:
                    fair_odds = OddsConverter.implied_to_american(prob)
                    table.add_row(
                        f"{thresh}",
                        f"{prob:.1%}",
                        f"{fair_odds:+d}"
                    )

            console.print(table)

        # If specific line provided, show edge analysis
        if line is not None:
            console.print(f"\n[bold]Edge Analysis for {line}:[/bold]")
            prob_over = prediction.prob_over(line)

            ev_calc = EVCalculator()
            edge = ev_calc.find_edge(prob_over, over_odds, under_odds)

            if edge:
                side = "OVER" if edge.is_over else "UNDER"
                color = "green" if edge.recommended_bet else "yellow"

                console.print(f"Model P(Over {line}): {prob_over:.1%}")
                console.print(f"Model P(Under {line}): {1-prob_over:.1%}")
                console.print(f"\nBest bet: [{color}]{side} {line}[/{color}]")
                console.print(f"Edge: {edge.edge_pct:.1f}%")
                console.print(f"EV: {edge.ev_pct:.1f}%")
                console.print(f"Kelly: {edge.kelly_fraction:.1%} (quarter: {edge.kelly_quarter:.1%})")

                if edge.recommended_bet:
                    console.print(f"\n[bold green]RECOMMENDED BET[/bold green]")
                else:
                    console.print(f"\n[yellow]Below threshold - no bet recommended[/yellow]")


@app.command()
def find_edges(
    min_ev: float = typer.Option(0.03, help="Minimum EV threshold"),
    min_games: int = typer.Option(10, help="Minimum games required"),
    limit: int = typer.Option(20, help="Max players to analyze")
):
    """Find +EV opportunities across all players."""
    import numpy as np
    from sqlalchemy import select, func

    from .config.constants import PropType, PROP_THRESHOLDS
    from .models.statistical.negative_binomial import NegativeBinomialModel
    from .models.base import PlayerContext
    from .betting.ev_calculator import EVCalculator

    console.print("[bold blue]Scanning for +EV opportunities...[/bold blue]\n")

    with get_session() as session:
        # Get players with enough games
        results = session.execute(
            select(
                Player,
                func.count(PlayerGameLog.id).label('game_count')
            )
            .outerjoin(PlayerGameLog)
            .where(Player.is_active == True)
            .group_by(Player.id)
            .having(func.count(PlayerGameLog.id) >= min_games)
            .order_by(func.count(PlayerGameLog.id).desc())
            .limit(limit)
        ).all()

        if not results:
            console.print("[yellow]No players with enough data[/yellow]")
            return

        ev_calc = EVCalculator(min_ev=min_ev)
        opportunities = []

        for player, game_count in results:
            logs = session.execute(
                select(PlayerGameLog)
                .where(PlayerGameLog.player_id == player.id)
                .order_by(PlayerGameLog.game_date.desc())
            ).scalars().all()

            # Check each prop type
            for prop_type in [PropType.POINTS, PropType.REBOUNDS, PropType.ASSISTS]:
                stat_attr = prop_type.value
                values = np.array([getattr(log, stat_attr) or 0 for log in logs])

                try:
                    model = NegativeBinomialModel(prop_type)
                    model.fit(values)
                    prediction = model.predict()

                    # Check common lines
                    for thresh in PROP_THRESHOLDS.get(prop_type, []):
                        prob_over = prediction.prob_over(thresh)

                        # Standard -110/-110 juice
                        edge = ev_calc.find_edge(prob_over, -110, -110)

                        if edge and edge.recommended_bet:
                            opportunities.append({
                                'player': player.name,
                                'prop': prop_type.value,
                                'line': thresh,
                                'side': 'OVER' if edge.is_over else 'UNDER',
                                'model_prob': edge.model_prob,
                                'ev': edge.expected_value,
                                'edge': edge.edge,
                                'kelly': edge.kelly_quarter
                            })
                except Exception:
                    continue

        if not opportunities:
            console.print("[yellow]No +EV opportunities found at current thresholds[/yellow]")
            return

        # Sort by EV
        opportunities.sort(key=lambda x: x['ev'], reverse=True)

        # Display
        table = Table(title=f"Top +EV Opportunities (min EV: {min_ev:.0%})")
        table.add_column("Player", style="cyan")
        table.add_column("Prop")
        table.add_column("Line", justify="right")
        table.add_column("Side")
        table.add_column("Prob", justify="right")
        table.add_column("EV", justify="right", style="green")
        table.add_column("Edge", justify="right")
        table.add_column("Kelly", justify="right")

        for opp in opportunities[:20]:
            table.add_row(
                opp['player'],
                opp['prop'],
                str(opp['line']),
                opp['side'],
                f"{opp['model_prob']:.1%}",
                f"{opp['ev']:.1%}",
                f"{opp['edge']:.1%}",
                f"{opp['kelly']:.1%}"
            )

        console.print(table)


@app.command()
def live_edges(
    min_ev: float = typer.Option(0.03, help="Minimum EV threshold"),
    prop_types: str = typer.Option("points,rebounds,assists", help="Comma-separated prop types")
):
    """Find +EV opportunities using live sportsbook odds."""
    import numpy as np
    from sqlalchemy import select

    from .config.constants import PropType
    from .models.statistical.negative_binomial import NegativeBinomialModel
    from .models.base import PlayerContext
    from .betting.ev_calculator import EVCalculator
    from .odds.odds_api import OddsAPIClient

    console.print("[bold blue]Fetching live odds from sportsbooks...[/bold blue]\n")

    try:
        client = OddsAPIClient()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    # Parse prop types
    prop_type_list = [p.strip() for p in prop_types.split(",")]
    api_markets = []
    for p in prop_type_list:
        if p == "points":
            api_markets.append("player_points")
        elif p == "rebounds":
            api_markets.append("player_rebounds")
        elif p == "assists":
            api_markets.append("player_assists")
        elif p == "threes":
            api_markets.append("player_threes")

    # Fetch props
    with console.status("[bold blue]Fetching player props..."):
        try:
            all_props = client.get_all_player_props(api_markets)
        except Exception as e:
            console.print(f"[red]Error fetching odds: {e}[/red]")
            return

    if not all_props:
        console.print("[yellow]No player props available right now[/yellow]")
        return

    console.print(f"Found {len(all_props)} prop lines from sportsbooks")
    console.print(f"API requests remaining: {client.requests_remaining}\n")

    # Get best lines per player/prop
    best_lines = client.get_best_lines(all_props)

    # Find edges
    ev_calc = EVCalculator(min_ev=min_ev)
    opportunities = []

    with get_session() as session:
        for key, lines_dict in best_lines.items():
            player_name, prop_type_str = key.split("|")

            # Map to PropType
            prop_map = {
                "points": PropType.POINTS,
                "rebounds": PropType.REBOUNDS,
                "assists": PropType.ASSISTS,
                "threes": PropType.THREES,
                "steals": PropType.STEALS,
                "blocks": PropType.BLOCKS,
            }
            prop_type = prop_map.get(prop_type_str)
            if not prop_type:
                continue

            # Find player in database
            player = session.execute(
                select(Player).where(Player.name.ilike(f"%{player_name}%"))
            ).scalar_one_or_none()

            if not player:
                continue

            # Get game logs
            logs = session.execute(
                select(PlayerGameLog)
                .where(PlayerGameLog.player_id == player.id)
                .order_by(PlayerGameLog.game_date.desc())
            ).scalars().all()

            if len(logs) < 5:
                continue

            # Get stat values
            stat_attr = prop_type.value
            if stat_attr == "threes":
                stat_attr = "fg3m"
            values = np.array([getattr(log, stat_attr) or 0 for log in logs])

            # Fit model
            try:
                model = NegativeBinomialModel(prop_type)
                model.fit(values)
                prediction = model.predict()
            except Exception:
                continue

            # Check each line
            for line, line_data in lines_dict.items():
                best_over = line_data["best_over"]
                best_under = line_data["best_under"]

                prob_over = prediction.prob_over(line)

                # Check edge with best available odds
                edge = ev_calc.find_edge(
                    prob_over,
                    best_over.over_odds,
                    best_under.under_odds
                )

                if edge and edge.recommended_bet:
                    if edge.is_over:
                        best_book = best_over.sportsbook
                        best_odds = best_over.over_odds
                    else:
                        best_book = best_under.sportsbook
                        best_odds = best_under.under_odds

                    opportunities.append({
                        'player': player_name,
                        'prop': prop_type_str,
                        'line': line,
                        'side': 'OVER' if edge.is_over else 'UNDER',
                        'model_prob': edge.model_prob,
                        'ev': edge.expected_value,
                        'edge': edge.edge,
                        'kelly': edge.kelly_quarter,
                        'odds': best_odds,
                        'book': best_book,
                        'game': f"{best_over.away_team} @ {best_over.home_team}"
                    })

    if not opportunities:
        console.print("[yellow]No +EV opportunities found with current thresholds[/yellow]")
        return

    # Sort by EV
    opportunities.sort(key=lambda x: x['ev'], reverse=True)

    # Display
    table = Table(title=f"Live +EV Opportunities (min EV: {min_ev:.0%})")
    table.add_column("Player", style="cyan")
    table.add_column("Game")
    table.add_column("Prop")
    table.add_column("Line", justify="right")
    table.add_column("Side")
    table.add_column("Odds", justify="right")
    table.add_column("Book", style="yellow")
    table.add_column("Prob", justify="right")
    table.add_column("EV", justify="right", style="green")
    table.add_column("Edge", justify="right")

    for opp in opportunities[:25]:
        table.add_row(
            opp['player'],
            opp['game'],
            opp['prop'],
            str(opp['line']),
            opp['side'],
            f"{opp['odds']:+d}",
            opp['book'],
            f"{opp['model_prob']:.0%}",
            f"{opp['ev']:.1%}",
            f"{opp['edge']:.1%}"
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(opportunities)} total opportunities[/dim]")


@app.command()
def fetch_odds():
    """Fetch and display current sportsbook odds."""
    from .odds.odds_api import OddsAPIClient

    console.print("[bold blue]Fetching live odds...[/bold blue]\n")

    try:
        client = OddsAPIClient()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    with console.status("[bold blue]Fetching player props..."):
        try:
            props = client.get_all_player_props()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if not props:
        console.print("[yellow]No props available[/yellow]")
        return

    # Group by game
    games = {}
    for prop in props:
        game_key = f"{prop.away_team} @ {prop.home_team}"
        if game_key not in games:
            games[game_key] = []
        games[game_key].append(prop)

    console.print(f"Found props for {len(games)} games")
    console.print(f"API requests remaining: {client.requests_remaining}\n")

    for game, game_props in games.items():
        console.print(f"[bold]{game}[/bold]")

        # Show sample of props
        shown = set()
        for prop in game_props[:15]:
            key = f"{prop.player_name}|{prop.prop_type}|{prop.line}"
            if key in shown:
                continue
            shown.add(key)

            console.print(
                f"  {prop.player_name} {prop.prop_type} {prop.line}: "
                f"O {prop.over_odds:+d} / U {prop.under_odds:+d} ({prop.sportsbook})"
            )
        console.print()


if __name__ == "__main__":
    app()
