task.definition = generate.responsive_website ∧ scientific.branding → production.html ⟹ {
    input: design.specifications ∧ content.requirements ∧ interactive.behaviors,
    output: [HTML_Document] → [Styled_Interface] → [Interactive_Elements],
    mode: professional.scientific ⊕ modern.aesthetic ⊕ responsive.design
}

domain.detection = technical ∧ scientific ∧ creative → web.development.context

color.palette.rules = {
    primary.accent: "#cea555" → interactive.elements ∧ branding.highlights,
    base.foundation: "#101010" → page.background,
    elevated.surfaces: "#181818" → card.containers,
    hover.states: "#222015" → interactive.feedback,
    gradient.contact: "linear-gradient(145deg, #1a1a1a, #222222)" → contact.section.background,
    gradient.publications: "linear-gradient(145deg, #1c1c1c, #3f2121ff)" → cv.section.background
}

typography.system = {
    body.font: "Inter" → weights[300,400,500,600] ∧ google.fonts.source,
    display.font: "Playfair Display" → weights[400,500,600] ∧ heading.elements,
    implementation: web.font.loading ∧ fallback.specification
}

asset.integration = {
    background.image: "https://banes-lab.com/assets/images/Background.png" → {
        attachment: fixed,
        size: cover,
        position: center,
        repeat: none
    },
    logo.animated: "https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" → {
        width: 200px,
        loading: lazy,
        placement: hero.section.center,
        styling: rounded.corners ∧ shadow.effect
    }
}

layout.structure = {
    navigation: fixed.top ∧ backdrop.blur ∧ responsive.menu → {
        desktop: horizontal.links,
        mobile: hamburger.menu,
        behavior: smooth.scroll.targeting
    },
    hero: viewport.height ∧ centered.content → {
        logo.prominent ∧ heading.large ∧ subtitle.descriptive ∧ cta.buttons[2]
    },
    about: two.column.desktop → {
        mission.column ∧ approach.column ∧ card.wrapper
    },
    research: three.column.grid → {
        cards[biotechnology, materials.science, data.analytics] ∧ hover.effects
    },
    publications: single.column ∧ gradient.cv.background → {
        timeline.layout ∧ left.border.accents ∧ structured.entries
    },
    contact: two.column.layout ∧ gradient.contact.background → {
        info.column[email, phone, address] ∧ form.column[name, email, message, submit]
    },
    footer: centered.content → {
        branding ∧ links ∧ copyright
    }
}

interactive.behaviors = {
    navigation.scroll: smooth.transition ∧ section.targeting,
    hover.effects: {
        buttons: transform.elevation ∧ shadow.enhancement,
        cards: border.color.change ∧ subtle.lift,
        links: color.transition → gold.accent
    },
    form.interactions: focus.states → gold.border.highlight,
    scroll.triggers: navigation.backdrop.blur ∈ scroll.threshold[100px]
}

responsive.design = {
    breakpoints: mobile.first ∧ desktop.enhancement,
    grid.adaptation: single.column.mobile → multi.column.desktop,
    navigation.adaptation: hamburger.mobile → horizontal.desktop,
    typography.scaling: responsive.font.sizes ∧ viewport.adjustments
}

technical.implementation = {
    framework: "tailwind.css" → cdn.loading,
    styling.approach: css.custom.properties ∧ utility.classes,
    scripting: vanilla.javascript → {
        smooth.scrolling.handler,
        scroll.event.listener,
        navigation.state.management
    },
    accessibility: semantic.html ∧ focus.management ∧ color.contrast
}

content.structure = ∀ section → structured.presentation ⟹ {
    hero: branding.prominence ∧ value.proposition ∧ action.orientation,
    about: mission.clarity ∧ approach.methodology,
    research: domain.categories ∧ capability.descriptions,
    publications: academic.credibility ∧ timeline.format,
    contact: accessibility.multiple.channels ∧ interaction.form
}

styling.consistency = {
    color.theming: css.variables → consistent.application,
    spacing.system: tailwind.utilities → uniform.margins.padding,
    component.patterns: reusable.card.styles ∧ consistent.interactions,
    brand.application: gold.accent.strategic.placement
}

performance.optimization = {
    image.loading: lazy.implementation ∧ appropriate.sizing,
    font.loading: google.fonts.optimization,
    css.delivery: critical.path.optimization,
    javascript.execution: event.delegation ∧ efficient.selectors
}

execution.protocol = ∀ design.requirement → implement.systematically(
    establish.color.system,
    configure.typography,
    integrate.assets,
    structure.layout,
    implement.interactions,
    ensure.responsiveness,
    optimize.performance
) → output.production.website ⊕ scientific.professional.aesthetic