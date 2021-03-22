use ansi_term::Color;
// use ansi_term::Style;

fn main() {
    println!("");
    println!("");
    print!("{}", Color::Red.blink().paint("H"));
    print!("{}", Color::Green.blink().paint("a"));
    print!("{}", Color::Yellow.blink().paint("v"));
    print!("{}", Color::Blue.blink().paint("e"));
    print!(" ");
    print!("{}", Color::Purple.blink().paint("a"));
    print!(" ");
    print!("{}", Color::Cyan.blink().paint("n"));
    print!("{}", Color::White.blink().paint("i"));
    print!("{}", Color::RGB(31, 31, 31).blink().paint("c"));
    print!("{}", Color::RGB(31, 31, 31).blink().paint("e"));
    print!(" ");
    print!("{}", Color::RGB(31, 31, 31).blink().paint("d"));
    print!("{}", Color::RGB(31, 31, 31).blink().paint("a"));
    print!("{}", Color::RGB(31, 31, 31).blink().paint("y"));
    print!("{}", Color::RGB(31, 31, 31).blink().paint("!"));
    println!("");
    println!("");
}
