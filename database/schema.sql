-- =========================================================================
-- NBA Probability Assistant - Staging Database Schema (SQL Server 2022/LocalDB)
-- Database: NBA_Prediction
-- =========================================================================

IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'NBA_Prediction')
BEGIN
    CREATE DATABASE NBA_Prediction;
END;
GO

USE NBA_Prediction;
GO

-- 1. Active Players Reference Table
IF OBJECT_ID('dbo.stg_players', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_players (
        PERSON_ID INT NOT NULL PRIMARY KEY,
        DISPLAY_FIRST_LAST NVARCHAR(100),
        DISPLAY_LAST_COMMA_FIRST NVARCHAR(100),
        ROSTERSTATUS INT,
        FROM_YEAR INT,
        TO_YEAR INT,
        PLAYERCODE NVARCHAR(50),
        TEAM_ID INT,
        TEAM_CITY NVARCHAR(50),
        TEAM_NAME NVARCHAR(50),
        TEAM_ABBREVIATION NVARCHAR(10),
        TEAM_CODE NVARCHAR(50),
        GAMES_PLAYED_FLAG NVARCHAR(5),
        OTHERLEAGUE_EXPERIENCE_CH INT,
        ingested_at DATETIME2 DEFAULT SYSDATETIME()
    );
END;
GO

-- 2. Player Game Logs
IF OBJECT_ID('dbo.stg_player_game_logs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_player_game_logs (
        SEASON_ID NVARCHAR(20),
        Player_ID INT NOT NULL,
        Game_ID NVARCHAR(20) NOT NULL,
        GAME_DATE NVARCHAR(20),
        MATCHUP NVARCHAR(50),
        WL NVARCHAR(5),
        MIN FLOAT,
        FGM FLOAT,
        FGA FLOAT,
        FG_PCT FLOAT,
        FG3M FLOAT,
        FG3A FLOAT,
        FG3_PCT FLOAT,
        FTM FLOAT,
        FTA FLOAT,
        FT_PCT FLOAT,
        OREB FLOAT,
        DREB FLOAT,
        REB FLOAT,
        AST FLOAT,
        STL FLOAT,
        BLK FLOAT,
        TOV FLOAT,
        PF FLOAT,
        PTS FLOAT,
        PLUS_MINUS FLOAT,
        VIDEO_AVAILABLE INT,
        SEASON NVARCHAR(20),
        PLAYER_NAME NVARCHAR(100),
        ingested_at DATETIME2 DEFAULT SYSDATETIME(),
        CONSTRAINT PK_stg_player_game_logs PRIMARY KEY (Player_ID, Game_ID)
    );
END;
GO

-- 3. Team Game Logs
IF OBJECT_ID('dbo.stg_team_game_logs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_team_game_logs (
        Team_ID INT NOT NULL,
        Game_ID NVARCHAR(20) NOT NULL,
        GAME_DATE NVARCHAR(20),
        MATCHUP NVARCHAR(50),
        WL NVARCHAR(5),
        W INT,
        L INT,
        W_PCT FLOAT,
        MIN FLOAT,
        FGM FLOAT,
        FGA FLOAT,
        FG_PCT FLOAT,
        FG3M FLOAT,
        FG3A FLOAT,
        FG3_PCT FLOAT,
        FTM FLOAT,
        FTA FLOAT,
        FT_PCT FLOAT,
        OREB FLOAT,
        DREB FLOAT,
        REB FLOAT,
        AST FLOAT,
        STL FLOAT,
        BLK FLOAT,
        TOV FLOAT,
        PF FLOAT,
        PTS FLOAT,
        SEASON NVARCHAR(20),
        TEAM_NAME NVARCHAR(50),
        ingested_at DATETIME2 DEFAULT SYSDATETIME(),
        CONSTRAINT PK_stg_team_game_logs PRIMARY KEY (Team_ID, Game_ID)
    );
END;
GO

-- 4. Traditional Box Score (V3)
IF OBJECT_ID('dbo.stg_boxscore_traditional', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_boxscore_traditional (
        gameId NVARCHAR(20) NOT NULL,
        teamId INT NOT NULL,
        personId INT NOT NULL,
        firstName NVARCHAR(50),
        familyName NVARCHAR(50),
        playerSlug NVARCHAR(50),
        position NVARCHAR(10),
        comment NVARCHAR(255),
        jerseyNum NVARCHAR(10),
        minutes NVARCHAR(20),
        fieldGoalsMade INT,
        fieldGoalsAttempted INT,
        fieldGoalsPercentage FLOAT,
        threePointersMade INT,
        threePointersAttempted INT,
        threePointersPercentage FLOAT,
        freeThrowsMade INT,
        freeThrowsAttempted INT,
        freeThrowsPercentage FLOAT,
        reboundsOffensive INT,
        reboundsDefensive INT,
        reboundsTotal INT,
        assists INT,
        steals INT,
        blocks INT,
        turnovers INT,
        foulsPersonal INT,
        points INT,
        plusMinusPoints FLOAT,
        ingested_at DATETIME2 DEFAULT SYSDATETIME(),
        CONSTRAINT PK_stg_boxscore_traditional PRIMARY KEY (gameId, personId)
    );
END;
GO

-- 5. Advanced Box Score (V3)
IF OBJECT_ID('dbo.stg_boxscore_advanced', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_boxscore_advanced (
        gameId NVARCHAR(20) NOT NULL,
        teamId INT NOT NULL,
        personId INT NOT NULL,
        firstName NVARCHAR(50),
        familyName NVARCHAR(50),
        position NVARCHAR(10),
        minutes NVARCHAR(20),
        offensiveRating FLOAT,
        defensiveRating FLOAT,
        netRating FLOAT,
        assistPercentage FLOAT,
        assistToTurnover FLOAT,
        offensiveReboundPercentage FLOAT,
        defensiveReboundPercentage FLOAT,
        reboundPercentage FLOAT,
        turnoverRatio FLOAT,
        effectiveFieldGoalPercentage FLOAT,
        trueShootingPercentage FLOAT,
        usagePercentage FLOAT,
        pace FLOAT,
        possessions FLOAT,
        PIE FLOAT,
        ingested_at DATETIME2 DEFAULT SYSDATETIME(),
        CONSTRAINT PK_stg_boxscore_advanced PRIMARY KEY (gameId, personId)
    );
END;
GO

-- 6. Team Rosters History
IF OBJECT_ID('dbo.stg_team_rosters', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_team_rosters (
        TeamID INT NOT NULL,
        PLAYER_ID INT NOT NULL,
        SEASON NVARCHAR(20) NOT NULL,
        PLAYER NVARCHAR(100),
        NICKNAME NVARCHAR(50),
        PLAYER_SLUG NVARCHAR(50),
        NUM NVARCHAR(10),
        POSITION NVARCHAR(10),
        HEIGHT NVARCHAR(10),
        WEIGHT NVARCHAR(10),
        BIRTH_DATE NVARCHAR(20),
        AGE FLOAT,
        EXP NVARCHAR(10),
        SCHOOL NVARCHAR(100),
        HOW_ACQUIRED NVARCHAR(100),
        TEAM_NAME NVARCHAR(50),
        ingested_at DATETIME2 DEFAULT SYSDATETIME(),
        CONSTRAINT PK_stg_team_rosters PRIMARY KEY (TeamID, PLAYER_ID, SEASON)
    );
END;
GO
